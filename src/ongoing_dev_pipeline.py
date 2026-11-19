"""Train and evaluate the enhanced LSTM using a chronological holdout."""

if __package__:
    from .training import main
else:
    from training import main


if __name__ == "__main__":
    main(enhanced=True)
