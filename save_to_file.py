import pickle
MODEL_PICKLE_FILENAME = 'Model.p'

def save_model(svc, orient, pix_per_cell, cell_per_block):
    print('Saving model to pickle file...')
    try:
        with open(MODEL_PICKLE_FILENAME, 'wb') as pfile:
            pickle.dump(
                {
                    'svc':svc,
                    'orient': orient,
                    'pix_per_cell': pix_per_cell,
                    'cell_per_block': cell_per_block,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', MODEL_PICKLE_FILENAME, ':', e)
        raise

    print('Model cached in pickle file:', MODEL_PICKLE_FILENAME)
