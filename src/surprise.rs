//! Surprise scoring — claude-hippo の差別化機能の核。

/// 単一 metric。0.0..=1.0 にクランプされる前提。
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct SurpriseComponents {
    /// 既存記憶ベクトル群からの cosine 距離 (mean) を 0..1 にマップ。
    pub embedding_outlier: f32,
    /// 応答の長さ・修正頻度等のヒューリスティック。
    pub engagement: f32,
    /// ユーザ明示マーク (importance param 由来)。
    pub explicit: f32,
    /// LLM の prediction loss (abyo-llm-probe 統合時に埋まる、それまで None)。
    pub prediction_loss: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
pub struct SurpriseWeights {
    pub w_outlier: f32,
    pub w_engagement: f32,
    pub w_explicit: f32,
    pub w_prediction: f32,
}

impl Default for SurpriseWeights {
    fn default() -> Self {
        Self {
            w_outlier: 0.4,
            w_engagement: 0.2,
            w_explicit: 0.1,
            w_prediction: 0.3,
        }
    }
}

impl SurpriseWeights {
    /// `"w_o,w_e,w_x,w_p"` 形式の文字列を SurpriseWeights に。
    /// すべて 0.0..=1.0、合計が 1.0 ± 1e-3 の範囲外なら Err。
    pub fn parse_csv(s: &str) -> std::result::Result<Self, String> {
        let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
        if parts.len() != 4 {
            return Err(format!(
                "expected 4 comma-separated weights (w_outlier,w_engagement,w_explicit,w_prediction), got {}",
                parts.len()
            ));
        }
        let mut vals = [0.0_f32; 4];
        for (i, p) in parts.iter().enumerate() {
            let v: f32 = p
                .parse()
                .map_err(|e| format!("weight #{} is not a number ({:?}): {e}", i + 1, p))?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!(
                    "weight #{} = {v} is out of range; must be in 0.0..=1.0",
                    i + 1
                ));
            }
            vals[i] = v;
        }
        let sum = vals.iter().sum::<f32>();
        if (sum - 1.0).abs() > 1e-3 {
            return Err(format!(
                "weights must sum to 1.0 (±1e-3), got {sum:.6} (= {} + {} + {} + {})",
                vals[0], vals[1], vals[2], vals[3]
            ));
        }
        Ok(Self {
            w_outlier: vals[0],
            w_engagement: vals[1],
            w_explicit: vals[2],
            w_prediction: vals[3],
        })
    }
}

/// 合成 surprise スコア。0.0..=1.0。prediction_loss が None の時は
/// w_prediction 分を w_outlier と engagement に按分する。
pub fn score(c: &SurpriseComponents, w: &SurpriseWeights) -> f32 {
    let outlier = c.embedding_outlier.clamp(0.0, 1.0);
    let engagement = c.engagement.clamp(0.0, 1.0);
    let explicit = c.explicit.clamp(0.0, 1.0);

    match c.prediction_loss {
        Some(p) => {
            let p = p.clamp(0.0, 1.0);
            outlier * w.w_outlier
                + engagement * w.w_engagement
                + explicit * w.w_explicit
                + p * w.w_prediction
        }
        None => {
            // w_prediction を outlier:engagement = 2:1 で按分
            let extra = w.w_prediction;
            let w_o = w.w_outlier + extra * (2.0 / 3.0);
            let w_e = w.w_engagement + extra * (1.0 / 3.0);
            outlier * w_o + engagement * w_e + explicit * w.w_explicit
        }
    }
    .clamp(0.0, 1.0)
}

/// 既存記憶ベクトル群との cosine 距離 mean を 0..1 へマップ。
/// すべて L2 normalized 前提。距離=1-cos sim、cos sim ∈ [-1,1] → 距離 ∈ [0,2]、
/// /2 で 0..1 にする。
pub fn embedding_outlier(query: &[f32], history: &[Vec<f32>]) -> f32 {
    if history.is_empty() {
        return 1.0; // 既存ゼロ → 完全に novel
    }
    let mut sum = 0.0_f32;
    for h in history {
        let cos = dot(query, h);
        let dist = (1.0 - cos).clamp(0.0, 2.0) / 2.0;
        sum += dist;
    }
    (sum / history.len() as f32).clamp(0.0, 1.0)
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// シンプルな engagement ヒューリスティック。content の長さ + tag 数の
/// 飽和関数。長文 + 多 tag は engagement 高い扱い。
pub fn engagement(content: &str, tag_count: usize) -> f32 {
    let len_score = (content.len() as f32 / 1000.0).tanh(); // 1000 chars で ~0.76
    let tag_score = (tag_count as f32 / 5.0).tanh(); // 5 tags で ~0.76
    (0.7 * len_score + 0.3 * tag_score).clamp(0.0, 1.0)
}

/// importance ∈ 0..=1 をそのまま explicit signal に使う。
pub fn explicit(importance: Option<f32>) -> f32 {
    importance.unwrap_or(0.0).clamp(0.0, 1.0)
}

/// 時間減衰 (forgetting curve)。Ebbinghaus 風の指数関数。
/// `age_days = 0` で 1.0、`half_life_days` で 0.5 に減衰。
pub fn decay(age_days: f32, half_life_days: f32) -> f32 {
    if half_life_days <= 0.0 {
        return 1.0;
    }
    (-(age_days.max(0.0)) * std::f32::consts::LN_2 / half_life_days).exp()
}

/// retrieval ranking 用合成スコア。
/// `relevance = 0.7 * cos_sim + 0.3 * surprise * max(decay(age), decay_floor)`
///
/// `decay_floor` ∈ [0,1] は forgetting curve の下限。0.0 だと v0.2 と同じ
/// 「365日越えで surprise·decay → 0、fresh chat の小 surprise·1 が old
/// high-surprise を demote」挙動。0.5 (v0.3 default) だと high-importance
/// Decision が無限時間後でも fresh low-importance chat に負けない。
pub fn ranking(
    cos_sim: f32,
    surprise: f32,
    age_days: f32,
    half_life_days: f32,
    decay_floor: f32,
) -> f32 {
    let d = decay(age_days, half_life_days).max(decay_floor.clamp(0.0, 1.0));
    let decayed = surprise * d;
    0.7 * cos_sim.clamp(0.0, 1.0) + 0.3 * decayed.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one() {
        let w = SurpriseWeights::default();
        let s = w.w_outlier + w.w_engagement + w.w_explicit + w.w_prediction;
        assert!((s - 1.0).abs() < 1e-6, "weights sum = {s}");
    }

    #[test]
    fn score_no_prediction_loss_redistributes() {
        let c = SurpriseComponents {
            embedding_outlier: 1.0,
            engagement: 1.0,
            explicit: 1.0,
            prediction_loss: None,
        };
        let s = score(&c, &SurpriseWeights::default());
        // すべて 1.0 + redistributed weights → 1.0
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn score_full_metrics_clamps_to_one() {
        let c = SurpriseComponents {
            embedding_outlier: 1.0,
            engagement: 1.0,
            explicit: 1.0,
            prediction_loss: Some(1.0),
        };
        let s = score(&c, &SurpriseWeights::default());
        assert!((s - 1.0).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn outlier_empty_history_is_novel() {
        let s = embedding_outlier(&[1.0, 0.0, 0.0], &[]);
        assert_eq!(s, 1.0);
    }

    #[test]
    fn outlier_identical_is_zero() {
        let v = vec![1.0_f32, 0.0, 0.0];
        let s = embedding_outlier(&v, std::slice::from_ref(&v));
        assert!(s.abs() < 1e-6, "got {s}");
    }

    #[test]
    fn outlier_orthogonal_is_half() {
        let q = vec![1.0_f32, 0.0, 0.0];
        let h = vec![vec![0.0_f32, 1.0, 0.0]];
        let s = embedding_outlier(&q, &h);
        assert!((s - 0.5).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn decay_at_half_life_is_half() {
        let d = decay(7.0, 7.0);
        assert!((d - 0.5).abs() < 1e-6, "got {d}");
    }

    #[test]
    fn decay_zero_age_is_one() {
        assert_eq!(decay(0.0, 7.0), 1.0);
    }

    #[test]
    fn parse_csv_default_roundtrip() {
        let w = SurpriseWeights::parse_csv("0.4,0.2,0.1,0.3").unwrap();
        let d = SurpriseWeights::default();
        assert!((w.w_outlier - d.w_outlier).abs() < 1e-6);
        assert!((w.w_engagement - d.w_engagement).abs() < 1e-6);
        assert!((w.w_explicit - d.w_explicit).abs() < 1e-6);
        assert!((w.w_prediction - d.w_prediction).abs() < 1e-6);
    }

    #[test]
    fn parse_csv_rejects_wrong_arity() {
        assert!(SurpriseWeights::parse_csv("0.5,0.5").is_err());
        assert!(SurpriseWeights::parse_csv("0.25,0.25,0.25,0.25,0.0").is_err());
    }

    #[test]
    fn parse_csv_rejects_out_of_range() {
        let r = SurpriseWeights::parse_csv("1.5,-0.5,0.0,0.0");
        assert!(r.is_err());
    }

    #[test]
    fn parse_csv_rejects_bad_sum() {
        let r = SurpriseWeights::parse_csv("0.5,0.5,0.5,0.5");
        assert!(r.is_err(), "sum=2.0 should be rejected");
        let r = SurpriseWeights::parse_csv("0.1,0.1,0.1,0.1");
        assert!(r.is_err(), "sum=0.4 should be rejected");
    }

    #[test]
    fn parse_csv_accepts_within_tolerance() {
        // sum = 1.0009 → within 1e-3
        let w = SurpriseWeights::parse_csv("0.2503,0.2503,0.2503,0.25").unwrap();
        let s = w.w_outlier + w.w_engagement + w.w_explicit + w.w_prediction;
        assert!((s - 1.0).abs() < 2e-3);
    }

    #[test]
    fn parse_csv_rejects_non_numeric() {
        assert!(SurpriseWeights::parse_csv("0.4,0.2,abc,0.4").is_err());
    }

    #[test]
    fn engagement_increases_with_length_and_tags() {
        let s_short = engagement("hi", 0);
        let s_long = engagement(&"x".repeat(1000), 5);
        assert!(s_long > s_short, "long={s_long} short={s_short}");
    }
}
