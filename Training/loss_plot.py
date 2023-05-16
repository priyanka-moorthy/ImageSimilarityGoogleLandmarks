import matplotlib.pyplot as plt
import config

epoch = config.EPOCHS
training_losses = [0.006189292762428522, 0.009910305961966515, 0.0065995450131595135, 0.012442849576473236]
validation_lossess = [0.013784639537334442, 0.01154748722910881, 0.010982529260218143, 0.01133654173463583]

plt.figure(figsize=(8,8))
plt.plot(training_losses, '-o', label='Train Losses')
plt.plot(validation_lossess, 'g-o', label='Valid Losses')
plt.legend()
plt.show()
