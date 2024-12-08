# Training Regimen for Chess AI Reinforcement Learning

This document provides a checklist-style guide for training a Chess AI using reinforcement learning. Each stage has specific hyperparameters and steps to help progressively improve model performance.

---

## Training Stages Overview

| Stage              | Cycles per Stage | Self-Play Games per Cycle | Epochs per Cycle  | Batch Size | MCTS Simulations per Move | AdamW Beta Values | Weight Decay |
|--------------------|------------------|--------------------------|-------------------|------------|---------------------------|-------------------|--------------|
| **Warm-Up**        | 5                | 100                      | 5                 | 64         | 75                        | β1=0.85, β2=0.98  | 1e-4         |
| **Main Training**  | 10               | 300                      | 10                | 128        | 200                       | β1=0.9, β2=0.999  | 1e-5         |
| **Fine-Tuning**    | 10               | 500                      | 20                | 256        | 400                       | β1=0.9, β2=0.999  | 1e-5         |

---

### Reasoning of Parameter Choices
- **Cycles per Stage**: Multiple cycles for each stage so that the model can adapt better.
- **Self-Play Games per Cycle**: Start with fewer games in the warm-up to capture basic patterns, increase for exploration in main training, and further increase for fine-tuning to capture nuanced strategies. This setup is on the lower resource side, balancing computational cost with performance.
- **Epochs per Cycle**: Fewer epochs in the first two stages to avoid overfitting early data and more epochs in fine-tuning to refine the model’s decisions.
- **Batch Size**: The batch size increases progressively, allowing faster adaptation in early stages and stable updates in later stages. This choice also considers computational resources.
- **Learning Rate**: The learning rate schedule increases during warm-up for larger updates and then gradually decreases in main training and fine-tuning for finer adjustments.
- **MCTS Simulations per Move**: A low start helps the model stabilize initially, with increases in later stages to provide more precise evaluations.
- **AdamW Beta Values**: Lower beta values in the beginning help the optimizer adapt quickly. Default beta values in the later stages provide stability.
- **Weight Decay**: Higher weight decay in warm-up helps prevent overfitting and stabilize early learning. Lower weight decay in later stages allows for finer adjustments.

---

## Stage 0: Untrained Model

**Objective**: Get weights for untrained model for later comparisons.

### Checklist
- [X] **Run Training**: Run `training_pipeline.py` with no data to get randomly initialized weights.

---

## Stage 1: Warm-Up

**Objective**: Stabilize the model with small updates to prevent large fluctuations in early training.

### Checklist
- **Total Cycles**: 5/5
1. **Configure Hyperparameters**:
   - [X] Set **MCTS Simulations per Move** to `75`.
   - [X] Set **Batch Size** to `64`.
   - [X] Change **Learning Rate Schedule** to warm-up. 
   - [X] Use **AdamW Optimizer** with the following settings:
     - [X] **Beta Values**: `β1=0.85`, `β2=0.98`
     - [X] **Weight Decay**: `1e-4`
2. **Play Self-Play Games**:
   - [X] Play `100` self-play games per cycle.
3. **Training**:
   - [X] Train the model with `5` epochs per cycle.
4. **Evaluate**:
   - [ ] Evaluate against previous checkpoints.

---

## Stage 2: Main Training

**Objective**: Broaden exploration and improve strategic patterns by using a high learning rate that gradually decays, along with more MCTS simulations.

### Checklist
- **Total Cycles**: 1/10
1. **Prepare Data and Backup**:
   - [X] Move the previous self-play data to a backup folder before generating new data.
2. **Configure Hyperparameters**:
   - [X] Increase **MCTS Simulations per Move** to `200`.
   - [X] Set **Batch Size** to `128`.
   - [X] Change **Learning Rate Schedule** Main. 
   - [X] Use **AdamW Optimizer** with the following settings:
     - [X] **Beta Values**: `β1=0.9`, `β2=0.999`
     - [X] **Weight Decay**: `1e-5`
3. **Play Self-Play Games**:
   - [ ] Play `300` self-play games per cycle.
4. **Training**:
   - [X] Train the model with `10` epochs per cycle.
5. **Evaluate**:
   - [ ] Evaluate against previous checkpoints.
---

## Stage 3: Fine-Tuning

**Objective**: Refine and polish the model with smaller updates and more detailed evaluations, converging on a highly refined play style.

### Checklist
- **Total Cycles**: 0/10
1. **Prepare Data and Backup**:
   - [ ] Move the previous self-play data to a backup folder before generating new data.
2. **Configure Hyperparameters**:
   - [ ] Increase **MCTS Simulations per Move** to `400`.
   - [ ] Set **Batch Size** to `256`.
   - [ ] Change **Learning Rate Schedule** to Finetuning. 
   - [ ] Use **AdamW Optimizer** with the following settings:
     - [ ] **Beta Values**: `β1=0.9`, `β2=0.999`
     - [ ] **Weight Decay**: `1e-5`
3. **Play Self-Play Games**:
   - [ ] Play `500` self-play games per cycle.
4. **Training**:
   - [ ] Train the model with `20` epochs per cycle.
5. **Evaluate**:
   - [ ] Evaluate against previous checkpoints.
---

