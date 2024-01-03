def evaluate(predictions, labels):
    print('******************************************')
    print('*********  EVALUATING PREDICTIONS ********')
    print('******************************************')
    print()

    # Calcularea altor metrici de evaluare
    accuracy = accuracy_score(labels, predictions)
    classification_rep = classification_report(labels, predictions)
    conf_mat = confusion_matrix(labels, predictions)

    print("Accuracy: {:.3f}".format(accuracy))
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", conf_mat)

    # Returnarea tuturor metricilor
    return accuracy, classification_rep, conf_mat


#accuracy, classification_rep, conf_matrix = evaluate(predictions_rf, df_test['label'])