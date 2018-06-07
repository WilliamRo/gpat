import tensorflow as tf
import core
from tframe import console
import model_lib as models
from core import mark_str as ms


def main(_):
  console.start('FC-LSTM task on GPAT')

  core.train_size = 8
  core.val_size = 2

  # Configurations
  th = core.th
  th.job_dir = core.from_gpat('fc_records')
  th.model = models.fc_lstm
  th.input_shape = [2000]
  input_dim = th.input_shape[0]

  th.fc_dims = [1000]
  th.rc_dims = [41]

  th.epoch = 1000
  th.learning_rate = 0.1
  th.batch_size = 1
  th.num_steps = 10

  # th.validate_cycle = 100
  th.validation_per_round = 2
  th.print_cycle = 1
  th.notify_when_reset = True
  th.early_stop = True
  th.idle_tol = 50

  # th.train = False
  th.overwrite = True
  th.save_model = True
  th.export_note = True
  th.summary = True
  th.monitor = False

  description = '_t8v2_0'
  th.mark = 'i{}_fc({})_rc({})'.format(
    input_dim, ms(th.fc_dims), ms(th.rc_dims))
  th.mark += description

  core.activate()


if __name__ == '__main__':
  tf.app.run()



