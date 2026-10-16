"""Run the whole Phase 2 pipeline: collect -> clean -> features -> eda.

    python data_scripts/run_all.py
"""
import clean
import collect
import eda
import features


def main():
    steps = [("1/4 collect", collect.main), ("2/4 clean", clean.main),
             ("3/4 features", features.main), ("4/4 eda", eda.main)]
    for label, fn in steps:
        print(f"\n===== {label} =====")
        fn()
    print("\nPhase 2 pipeline finished. See figures/ and logs/.")


if __name__ == "__main__":
    import sys
    sys.argv = sys.argv[:1]   # collect.main() parses CLI args; run_all takes none
    main()
