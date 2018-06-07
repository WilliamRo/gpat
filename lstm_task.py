import tensorflow as tf
import core
from tframe import console
import model_lib as models
from core import mark_str as ms


def main(_):
  console.start('LSTM task on GPAT')

  core.train_size = 50
  core.val_size = 4
  core.batches_per_epoch = 1000

  # Configurations
  th = core.th
  th.job_dir = core.from_gpat('lstm_task')
  th.model = models.lstm0
  th.input_shape = [1000]
  input_dim = th.input_shape[0]

  # th.rc_dims = [1000, 41]
  th.rc_dims = [500]

  th.fc_dims = []

  th.epoch = 1000
  th.learning_rate = 0.1
  th.batch_size = 20
  th.num_steps = 4

  th.validate_cycle = 200
  # th.validation_per_round = 10
  th.print_cycle = 1
  th.notify_when_reset = True
  th.early_stop = True
  th.idle_tol = 50

  # th.train = False
  # th.overwrite = True
  th.save_model = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = '_t{}v{}lr{}'.format(
    core.train_size, core.val_size, th.learning_rate)
  th.mark = 'i{}_rc({})_s{}_bs{}'.format(
    input_dim, ms(th.rc_dims), th.num_steps, th.batch_size)
  if len(th.fc_dims) > 0: th.mark += '_fc({})'.format(ms(th.fc_dims))
  th.mark += description

  core.activate()


if __name__ == '__main__':
  tf.app.run()



