def average_predictions(cross_val_errors, filenames, num_test_examples=138):
    average = [0] * num_test_examples

    total_error = 0
    for error in cross_val_errors:
        total_error = total_error + error

    weights = [0] * len(cross_val_errors)
    for (idx, error) in enumerate(cross_val_errors):
        weights[idx] = total_error - error

    total_weight = sum(weights)
    weights = [x / float(total_weight) for x in weights]

    for w_idx, file in enumerate(filenames):
        with open(file, 'r') as pred:
            pred.readline()

            for i in range(num_test_examples):
                status = pred.readline().split(',')[1]

                average[i] = average[i] + weights[w_idx] * float(status)


    with open('./final_submission.csv', 'w') as out:
        out.write("ID,Prediction\n")
        for i in range(1, num_test_examples + 1):
            out.write(str(i) + "," + str(average[i - 1]) + "\n")
