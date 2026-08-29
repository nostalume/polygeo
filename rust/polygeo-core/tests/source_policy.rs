use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(root: &Path, output: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(root).expect("source directory must be readable") {
        let path = entry.expect("source entry must be readable").path();
        if path.is_dir() {
            rust_sources(&path, output);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            output.push(path);
        }
    }
}

#[test]
fn production_rust_has_no_allow_attributes() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core crate belongs to the Rust workspace");
    let mut violations = Vec::new();
    for member in ["polygeo-core", "polygeo-py"] {
        let mut sources = Vec::new();
        rust_sources(&workspace.join(member).join("src"), &mut sources);
        for source in sources {
            let contents = fs::read_to_string(&source).expect("Rust source must be UTF-8");
            for (line, text) in contents.lines().enumerate() {
                if text.contains("#[allow(") {
                    violations.push(format!(
                        "{}:{}: {}",
                        source.display(),
                        line + 1,
                        text.trim()
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "production allow attributes are forbidden:\n{}",
        violations.join("\n")
    );
}
