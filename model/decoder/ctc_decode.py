def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):

  input_shape = shape(y_pred)
  samples, steps = input_shape[0], input_shape[1]
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
  input_length = math_ops.cast(input_length, dtypes_module.int32)

  if greedy:
    (decoded, log_prob) = ctc.ctc_greedy_decoder(
        inputs=y_pred, sequence_length=input_length)
  else:
    (decoded, log_prob) = ctc.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths)
  decoded_dense = [
      sparse_ops.sparse_to_dense(
          st.indices, (samples, steps), st.values, default_value=-1)
      for st in decoded
  ]
  return (decoded_dense, log_prob)