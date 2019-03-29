import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')

state = tf.Variable(0, name="counter")

one = tf.constant(1)
addedOne = tf.add(state, one)
update = tf.assign(state, addedOne)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    print(sess.run(hello))

    a = tf.constant(10)
    b = tf.constant(32)
    print(sess.run(a+b))

    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.], [2.]])
    product = tf.matmul(matrix1, matrix2)
    print(sess.run(product))

    # x = tf.Variable([1.0, 2.0])
    # a = tf.constant([3.0, 3.0])
    # x.initializer.run()
    # sub = tf.sub(x, a)
    # print(sub.eval())

    sess.run(init_op)
    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.multiply(input1, intermed)
    result = sess.run([mul, intermed])
    print(result)

    input1 = tf.placeholder(tf.dtypes.float32)
    input2 = tf.placeholder(tf.dtypes.float32)
    output = tf.multiply(input1, input2)
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))
