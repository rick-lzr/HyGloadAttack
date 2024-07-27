# HyGloadAttack
HyGloadAttack code implementation

This repository contains source code for the research work described in our Neural Networks paper:
[HyGloadAttack: Hard-label black-box textual adversarial attacks via hybrid optimization](https://www.sciencedirect.com/science/article/abs/pii/S089360802400385X)

This method is very efficient, requiring only over 300 QRS on the MR dataset to achieve extremely high performance.  The other attacked datasets all only require an average of 1000 to 2000 QRS to complete highly efficient attacks.
## Implementation Instruction
- Fork the repository https://github.com/RishabhMaheshwary/hard-label-attack and follow its instruction to install the environment
- First, run the code from **hard-label-attack**. Then simply run the run.sh script
- Note: The paths for some dependency files are hardcoded in the code and need to be manually changed.

## Attack Adversarial Training Model
We use code from the "dne" repository https://github.com/dugu9sword/dne to implement adversarial training.

## Acknowledgement
We thank the authors of https://github.com/RishabhMaheshwary/hard-label-attack for sharing their code.

## If you find our repository helpful, consider citing our work.
```
@article{DBLP:journals/nn/LiuXLYLZX24,
  author       = {Zhaorong Liu and
                  Xi Xiong and
                  Yuanyuan Li and
                  Yan Yu and
                  Jiazhong Lu and
                  Shuai Zhang and
                  Fei Xiong},
  title        = {HyGloadAttack: Hard-label black-box textual adversarial attacks via
                  hybrid optimization},
  journal      = {Neural Networks},
  volume       = {178},
  pages        = {106461},
  year         = {2024},
}
```
