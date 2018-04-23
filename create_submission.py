import pandas as pd
from global_variables import SAMPLE_SUBMISSION

def create_submission_from_prediction(csv_file, out_fn):
    submission = pd.read_csv(SAMPLE_SUBMISSION, index_col=0)


    test_prediction_df = pd.read_csv(csv_file,index_col=0)
    fns = [r[5:-4] for r in test_prediction_df.index]

    res = {}

    for fn in fns:
        a = test_prediction_df.loc['test/' + fn + '.jpg']
        label = a.index[a.values.argmax()]
        res[fn] = label


    s = submission.to_dict()

    for item in res:
        s['predicted'][int(item)] = int(res[item])

    df = pd.DataFrame.from_dict(s)
    df.to_csv(out_fn,index_label='id')