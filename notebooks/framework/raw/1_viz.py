print('Loss at step %d: %f' % (step, l))
print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels[:train_subset, :]))
print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
