def evaluate(predictions, labels):
    print('******************************************')
    print('*********  EVALUATING PREDICTIONS ********')
    print('******************************************')
    print()

    accuracy = accuracy_score(labels, predictions)
    classification_rep = classification_report(labels, predictions)

    print("Accuracy: {:.3f}".format(accuracy))
    print("\nClassification Report:\n", classification_rep)