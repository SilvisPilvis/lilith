use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <labels.json>", args[0]);
        eprintln!("  Analyzes preference score distribution in a JSON file");
        eprintln!("  JSON format: {{\"image_name.jpg\": 0.85, ...}}");
        std::process::exit(1);
    }

    let path = &args[1];

    if !Path::new(path).exists() {
        eprintln!("Error: File not found: {}", path);
        std::process::exit(1);
    }

    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            std::process::exit(1);
        }
    };

    let labels: HashMap<String, f32> = match serde_json::from_str(&content) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Error parsing JSON: {}", e);
            std::process::exit(1);
        }
    };

    let total = labels.len();
    if total == 0 {
        eprintln!("Error: No labels found in file");
        std::process::exit(1);
    }

    println!("Loaded {} labels from {}", total, path);
    println!();

    let mut buckets: [usize; 10] = [0; 10];

    for (_, &score) in &labels {
        let clamped = score.clamp(0.0, 1.0);
        let bucket = ((clamped * 10.0).floor() as usize).min(9);
        buckets[bucket] += 1;
    }

    let max_count = *buckets.iter().max().unwrap_or(&1);
    let bar_width = 40;

    println!("Preference Score Distribution");
    println!("");
    println!("{:<12} {:>8} {:>8}  {}", "Bucket", "Count", "Ratio", "Bar");
    println!("{}", "-".repeat(60));

    for (i, &count) in buckets.iter().enumerate() {
        let bucket_start = i as f32 / 10.0;
        let bucket_end = (i + 1) as f32 / 10.0;
        let bucket_label = format!("{:.1}-{:.1}", bucket_start, bucket_end);

        let ratio = count as f64 / total as f64;
        let bar_len = if max_count > 0 {
            (count as f64 / max_count as f64 * bar_width as f64).round() as usize
        } else {
            0
        };

        let bar = "█".repeat(bar_len);
        let empty = "░".repeat(bar_width - bar_len);

        println!(
            "{:<12} {:>8} {:>7.2}%  {}{}",
            bucket_label,
            count,
            ratio * 100.0,
            bar,
            empty
        );
    }

    println!("{}", "-".repeat(60));
    println!("{:<12} {:>8} {:>7.2}%", "Total", total, 100.0);
    println!();

    let ideal_per_bucket = total as f64 / 10.0;
    let mut max_deviation = 0.0_f64;
    let mut deviations = Vec::new();

    for (i, &count) in buckets.iter().enumerate() {
        let actual = count as f64;
        let deviation = ((actual - ideal_per_bucket).abs() / ideal_per_bucket) * 100.0;
        deviations.push((i, deviation));
        if deviation > max_deviation {
            max_deviation = deviation;
        }
    }

    deviations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Ideal count per bucket: {:.1}", ideal_per_bucket);
    println!("Max deviation: {:.1}%", max_deviation);
    println!();

    if max_deviation < 10.0 {
        println!("✓ Distribution is well balanced");
    } else if max_deviation < 30.0 {
        println!("~ Distribution is moderately imbalanced");
    } else {
        println!("✗ Distribution is heavily imbalanced");
    }

    println!();
    println!("Buckets by deviation from ideal:");
    for (bucket, dev) in deviations.iter().take(5) {
        let bucket_start = *bucket as f32 / 10.0;
        let bucket_end = (*bucket + 1) as f32 / 10.0;
        println!(
            "  {:.1}-{:.1}: {:.1}% deviation",
            bucket_start, bucket_end, dev
        );
    }

    let mut sorted_scores: Vec<f32> = labels.values().copied().collect();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted_scores.first().unwrap_or(&0.0);
    let max = sorted_scores.last().unwrap_or(&1.0);
    let median = if sorted_scores.len() % 2 == 0 {
        let mid = sorted_scores.len() / 2;
        (sorted_scores[mid - 1] + sorted_scores[mid]) / 2.0
    } else {
        sorted_scores[sorted_scores.len() / 2]
    };
    let mean: f32 = sorted_scores.iter().sum::<f32>() / sorted_scores.len() as f32;

    println!();
    println!("Statistics");
    println!("{}", "-".repeat(60));
    println!("Min:    {:.4}", min);
    println!("Max:    {:.4}", max);
    println!("Mean:   {:.4}", mean);
    println!("Median: {:.4}", median);
}
