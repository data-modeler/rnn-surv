'''
    Losses
    -----

    The loss functions to use in the RNN-SURV model.    
'''
import tensorflow.keras.backend as K


def loss1(pad_token):
    
    def loss(y_true, y_pred):

        mask = K.cast(K.not_equal(y_true, pad_token), K.floatx())
        return K.binary_crossentropy(y_true * mask, y_pred * mask)

    return loss

def loss2(mask_layer, pad_token, flip_i):

    def loss(y_true, y_pred):
        ''' y_true are the sequence lengths, y_pred is the risk calculation'''
        # create mask for only using observations where the event occurred
        # to calculate the loss
        m1 = K.cast(K.equal(mask_layer, 1.), K.floatx())
        y_shape = K.shape(y_true) # (batch size, 1)

        phi_mask = K.reshape(K.sum(m1, axis=1), y_shape) # masks out censored observations

        ### mask where Y_i >= Y_j
        y_eye = K.flatten(y_true) * flip_i  # square matrix where values of y are in the columns
        y_eye_t = K.transpose(y_eye)

        y_mask = K.greater_equal(y_eye, y_eye_t) # square matrix where Yi >= Yj
        y_mask = K.cast(y_mask, K.floatx())
        y_mask = y_mask * flip_i
        
        mask = y_mask * phi_mask

        # to calculate the differences, this will create a matrix of diffs
        mat = y_pred * flip_i 

        r_diffs = K.transpose(mat) - mat # r_j - r_i
        r_diffs = r_diffs * mask
        r_diffs = 1 + (K.log(K.sigmoid(r_diffs)) / K.log(K.constant(2)))

        return -1 * K.sum(K.sum(r_diffs)) / K.sum(K.sum(y_mask))

    return loss

