use anyhow::{Result, bail};

fn main() -> Result<()> {
    bail!(
        "pt_to_mpk is temporarily disabled for the new residual/attention model layout. Use native Burn checkpoints for now or update this converter with the new state-dict mapping."
    )
}
