Experiment 1 — Baseline CNN (from scratch)
- Architecture: 3x Conv2D+MaxPool blocks, Dense(128), Dropout(0.3)
- Trained: 5 epochs (EarlyStopping, patience=3), stopped at epoch 2 best weights
- Train accuracy: 0.61, Val accuracy: 0.60
- QWK: 0.4972
- Per-class recall: Grade 0: 0.749, Grade 1: 0.691, Grade 2: 0.400, Grade 3: 0.103, Grade 4: 0.250
- Key finding: severe train/val recall gap on minority classes (Grade 3
  especially) despite class_weight balancing — motivates transfer learning
  in next phase.

  