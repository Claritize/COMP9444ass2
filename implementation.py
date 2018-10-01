import tensorflow as tf

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    #print(review+"\n")
    
    review = review.lower()

    #review = review.replace(",", " ")
    #review = review.replace("'", "")
    #review = review.replace("\"", "")
    #review = review.replace(":", "")
    #review = review.replace("/", "")
    #review = review.replace("_", " ")
    #review = review.replace("(", "")
    #review = review.replace(")", "")
    review = review.replace("/><br", "")
    review = review.replace("/>", "")
    review = review.replace("<br", "")
    review = review.replace(")", "")
    review = review.replace("<", "")
    review = review.replace(">", "")
    review = review.replace("*", "")
    review = review.replace("vs.", "versus")

    review = " ".join(review.split())

    return review

def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(128)

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    #placeholder for data
    input_data = tf.placeholder(tf.float32, name="input_data", shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE])
    #placeholder for labels
    labels = tf.placeholder(tf.float32, name="labels", shape=[BATCH_SIZE, 2])
    
    #dropout probability
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

    #first convolution layer
    conv = tf.layers.conv1d(input_data, 32, 3, activation=tf.nn.relu)

    #pooling layer
    pool = tf.layers.max_pooling1d(conv, 2, 1)

    batch = tf.layers.batch_normalization(pool)

    #initialsing LSTM layer
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(1)])
    state = stacked_lstm.zero_state(BATCH_SIZE, tf.float32)

    #dropout for LSTM layer
    drop0 = tf.contrib.rnn.DropoutWrapper(stacked_lstm, output_keep_prob=dropout_keep_prob)

    #running the LSTM layer
    outputs, state = tf.nn.dynamic_rnn(drop0, batch, dtype=tf.float32)

    #dense layer to produce logits
    dense = tf.layers.dense(outputs[:,-1], 100, activation=tf.nn.sigmoid)
    drop1 = tf.layers.dropout(dense, rate=(1-dropout_keep_prob))
    logits = tf.layers.dense(drop1, 2, activation=None)

    #generating error rate and loss calculations
    error = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
    loss = tf.reduce_sum(error, name="loss")
    #loss = tf.losses.softmax_cross_entropy(labels, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    #generatring accuracy and correctness
    preds = tf.nn.softmax(logits)
    correct = tf.equal(tf.argmax(preds, axis=1), tf.argmax(labels, axis=1))
    Accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
