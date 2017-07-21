Output of the model MLP:

Learning rate 0.1:
INFO:root:Epoch[0] Batch [100]	Speed: 41885.53 samples/sec	accuracy=0.845050
INFO:root:Epoch[0] Batch [200]	Speed: 41167.69 samples/sec	accuracy=0.927200
INFO:root:Epoch[0] Batch [300]	Speed: 36100.69 samples/sec	accuracy=0.939200
INFO:root:Epoch[0] Batch [400]	Speed: 37398.12 samples/sec	accuracy=0.945100
INFO:root:Epoch[0] Batch [500]	Speed: 35500.27 samples/sec	accuracy=0.951100
INFO:root:Epoch[0] Train-accuracy=0.957172
INFO:root:Epoch[0] Time cost=1.798
INFO:root:Epoch[0] Validation-accuracy=0.957700

learning rate 1:
INFO:root:Epoch[0] Batch [100]	Speed: 45857.68 samples/sec	accuracy=0.792871
INFO:root:Epoch[0] Batch [200]	Speed: 44661.40 samples/sec	accuracy=0.909600
INFO:root:Epoch[0] Batch [300]	Speed: 41232.52 samples/sec	accuracy=0.928900
INFO:root:Epoch[0] Batch [400]	Speed: 43398.45 samples/sec	accuracy=0.937600
INFO:root:Epoch[0] Batch [500]	Speed: 43790.52 samples/sec	accuracy=0.943000
INFO:root:Epoch[0] Train-accuracy=0.942929
INFO:root:Epoch[0] Time cost=1.380
INFO:root:Epoch[0] Validation-accuracy=0.952500

Output of model conv:

learning rate 0.1:
INFO:root:Epoch[0] Batch [100]	Speed: 1423.92 samples/sec	accuracy=0.112277
INFO:root:Epoch[0] Batch [200]	Speed: 1417.03 samples/sec	accuracy=0.108700
INFO:root:Epoch[0] Batch [300]	Speed: 1429.30 samples/sec	accuracy=0.240900
INFO:root:Epoch[0] Batch [400]	Speed: 1467.34 samples/sec	accuracy=0.625000
INFO:root:Epoch[0] Batch [500]	Speed: 1471.50 samples/sec	accuracy=0.783600
INFO:root:Epoch[0] Train-accuracy=0.845455
INFO:root:Epoch[0] Time cost=41.573
INFO:root:Epoch[0] Validation-accuracy=0.873100

learning rate 1:
INFO:root:Epoch[0] Batch [100]	Speed: 1469.99 samples/sec	accuracy=0.513465
INFO:root:Epoch[0] Batch [200]	Speed: 1468.92 samples/sec	accuracy=0.877600
INFO:root:Epoch[0] Batch [300]	Speed: 1472.09 samples/sec	accuracy=0.910700
INFO:root:Epoch[0] Batch [400]	Speed: 1462.19 samples/sec	accuracy=0.918200
INFO:root:Epoch[0] Batch [500]	Speed: 1444.73 samples/sec	accuracy=0.922800
INFO:root:Epoch[0] Train-accuracy=0.927879
INFO:root:Epoch[0] Time cost=41.265
INFO:root:Epoch[0] Validation-accuracy=0.937800

Future improvement:
1. inception layer
2. Hyperparameter tuning
3. AlexNet

Comparison Table:

Learning rate 0.1:

Comparison | MLP | Conv |

|Train Acc(5 epoch) | 0.957172 | 0.845455 |

|Val Acc(5 epoch) | 0.952500 | 0.873100 |

|CPU Epoch time | 1.798 | 41.573 |