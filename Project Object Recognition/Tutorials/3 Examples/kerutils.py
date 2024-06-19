from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import Callback
from matplotlib import pyplot as plt
import numpy as np
import warnings
from dlutils import *
# fixed random seed for reproducibility
np.random.seed(0)

class FitMonitor(Callback):
    def __init__(self, **opt):
        super(Callback, self).__init__()
        #Callback.__init__(self)
        self.thresh = opt.get('thresh', 0.02) # max difference between acc and val_acc for saving model
        self.minacc = opt.get('minacc', 0.99) # minimal accuracy for model saving
        self.best_acc = self.minacc
        self.filename = opt.get('filename', None)
        self.verbose = opt.get('verbose', 1)
        self.checkpoint = None
        self.stop_file = 'stop_training_file.keras'
        self.pause_file = 'pause_training_file.keras'
        self.hist = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': []}

    def on_epoch_begin(self, epoch, logs={}):
        #print "epoch begin:", epoch
        self.curr_epoch = epoch

    def on_train_begin(self, logs={}):
        "This is the point where training is starting. Good place to put all initializations"
        self.start_time = datetime.datetime.now()
        t = datetime.datetime.strftime(self.start_time, '%Y-%m-%d %H:%M:%S')
        print ("Train begin:", t)
        print ("Stop file: %s (create this file to stop training gracefully)" % self.stop_file)
        print ("Pause file: %s (create this file to pause training and view graphs)" % self.pause_file)
        self.print_params()
        self.progress = 0
        self.max_acc = 0
        self.max_val_acc = -1
        self.max_loss = 0
        self.max_val_loss = -1
        self.max_acc_epoch = -1
        self.max_val_acc_epoch = -1

    def on_train_end(self, logs={}):
        "This is the point where training is ending. Good place to summarize calculations"
        self.end_time = datetime.datetime.now()
        t = datetime.datetime.strftime(self.end_time, '%Y-%m-%d %H:%M:%S')
        print ("Train end:", t)
        dt = self.end_time - self.start_time
        if self.verbose:
            h = dt.total_seconds()/3600.0
            print ("Total run time: %.3f hours" % (h,))
            print ("max_acc = %f  epoch = %d" % (self.max_acc, self.max_acc_epoch))
            print ("max_val_acc = %f  epoch = %d" % (self.max_val_acc, self.max_val_acc_epoch))
        if self.filename:
            if self.checkpoint:
                print ("Best model saved in file:", self.filename)
                print ("Checkpoint: epoch=%d, acc=%.6f, val_acc=%.6f" % self.checkpoint)
            else:
                print ("No checkpoint model found.")
                #print "Saving the last state:", self.filename
                #self.model.save(self.filename)

    def on_batch_end(self, batch, logs={}):
        #print "epoch=%d, batch=%s, acc=%f" % (self.curr_epoch, batch, logs.get('acc'))
        #self.probe(logs)
        if os.path.exists(self.pause_file):
            os.remove(self.pause_file)
            self.plot_hist()

    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get('acc')
        val_acc = logs.get('val_acc', -1)
        loss = logs.get('loss')
        val_loss = logs.get('val_loss', -1)
        self.hist['acc'].append(acc)
        self.hist['loss'].append(loss)
        if val_acc != -1:
            self.hist['val_acc'].append(val_acc)
            self.hist['val_loss'].append(val_loss)

        p = int(epoch / (self.params['epochs'] / 100.0))
        if p > self.progress:
            sys.stdout.write('.')
            if p%5 == 0:
                dt = datetime.datetime.now() - self.start_time
                h = dt.total_seconds()/3600.0
                fmt = '%02d%% epoch=%d, acc=%f, loss=%f, val_acc=%f, val_loss=%f, time=%.3f hours\n'
                vals = (p,    epoch,    acc,    loss,    val_acc,    val_loss,    h)
                sys.stdout.write(fmt % vals)
            sys.stdout.flush()
            self.progress = p
        if epoch == self.params['epochs'] - 1:
            sys.stdout.write(' %d%% epoch=%d acc=%f loss=%f\n' % (p, epoch, acc, loss))

        self.probe(logs)

    def probe(self, logs):
        epoch = self.curr_epoch
        acc = logs.get('acc')
        val_acc = logs.get('val_acc', -1)
        loss = logs.get('loss')
        val_loss = logs.get('val_loss', -1)
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
            self.model.stop_training = True

        if os.path.exists(self.pause_file):
            os.remove(self.pause_file)
            self.plot_hist()

        if val_acc > self.max_val_acc:
            self.max_val_acc = val_acc
            self.max_val_acc_epoch = epoch

        if acc > self.max_acc:
            self.max_acc = acc
            self.max_acc_epoch = epoch
            if self.filename != None:
                if acc > self.best_acc and (val_acc == -1 or abs(val_acc - acc) <= self.thresh):
                    print ("\nSaving model to %s: epoch=%d, acc=%f, val_acc=%f" % (self.filename, epoch, acc, val_acc))
                    self.model.save(self.filename)
                    self.checkpoint = (epoch, acc, val_acc)
                    self.best_acc = acc

        self.max_loss = max(self.max_loss, loss)
        self.max_val_loss = max(self.max_val_loss, val_loss)

    def plot_hist(self):
        #loss, acc = self.model.evaluate(X_train, Y_train, verbose=0)
        #print "Training: accuracy   = %.6f loss = %.6f" % (acc, loss)
        #X = m.validation_data[0]
        #Y = m.validation_data[1]
        #loss, acc = self.model.evaluate(X, Y))
        #print "Validation: accuracy = %.6f loss = %.6f" % (acc, loss)
        # Accuracy history graph
        plt.plot(self.hist['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if self.hist['val_acc']:
            plt.plot(self.hist['val_acc'])
            leg = plt.legend(['train', 'validation'], loc='best')
            plt.setp(leg.get_lines(), linewidth=3.0)
        plt.show()
        plt.plot(self.hist['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        if self.hist['val_loss']:
            plt.plot(self.hist['val_loss'])
            leg = plt.legend(['train', 'validation'], loc='best')
            plt.setp(leg.get_lines(), linewidth=3.0)
        plt.show()

    def print_params(self):
        for key in sorted(self.params.keys()):
            print ("%s = %s" % (key, self.params[key]))

#-------------------------------------------------------------

class BreakOnMonitor(Callback):
    def __init__(self, monitor='acc', value=0.8, epoch_limit=30, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.epoch_limit = epoch_limit
        self.verbose = verbose
        self.max_value = 0
        self.stop_file = 'stop_training_file.keras'

    def on_train_begin(self, logs={}):
        print ("Stop file: %s (create this file to stop training gracefully)" % self.stop_file)

    def on_epoch_end(self, epoch, logs={}):
        curr_acc = logs.get(self.monitor)
        if curr_acc is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if curr_acc > self.max_value:
            self.max_value = curr_acc

        if epoch > self.epoch_limit and self.max_value < self.value:
            if self.verbose > 0:
                print ("\nEARLY STOPPING: epoch=%d ; No monitor progress" % epoch)
            self.model.stop_training = True

        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
            self.model.stop_training = True

#-------------------------------------------------------------

# h - history object returned by Keras model method
def show_scores(model, h, X_train, Y_train, X_test, Y_test):
    loss, acc = model.evaluate(X_train, Y_train, verbose=0)
    print ("Training: accuracy   = %.6f loss = %.6f" % (acc, loss))
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print ("Validation: accuracy = %.6f loss = %.6f" % (acc, loss))
    if 'val_acc' in h.history:
        print ("Over fitting score   = %.6f" % over_fitting_score(h))
        print ("Under fitting score  = %.6f" % under_fitting_score(h))
    print ("Params count:", model.count_params())
    print ("stop epoch =", max(h.epoch))
    print ("nb_epoch =", h.params['epochs'])
    #print ("batch_size =", h.params['batch_size'])
    #print ("nb_sample =", h.params['samples'])
    view_acc(h)
    id = model.name[-1]
    plt.savefig(model.name + '_acc_graph.png')
    plt.show()
    view_loss(h)
    plt.savefig(model.name + '_loss_graph.png')
    plt.show()

def view_acc(h):
    # Accuracy history graph
    plt.plot(h.history['accuracy'])
    if 'val_accuracy' in h.history:
        plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    leg = plt.legend(['train', 'validation'], loc='best')
    plt.setp(leg.get_lines(), linewidth=3.0)

def view_loss(h):
    # Loss history graph
    plt.plot(h.history['loss'])
    if 'val_loss' in h.history:
        plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    leg = plt.legend(['train', 'validation'], loc='best')
    plt.setp(leg.get_lines(), linewidth=3.0)

# http://machinelearningmastery.com/improve-deep-learning-performance
def over_fitting_score(h):
    gap = []
    n = len(h.epoch)
    for i in h.epoch:
        acc = h.history['acc'][i]
        val_acc = h.history['val_accuracy'][i]
        # late gaps get higher weight ..
        gap.append( i * abs(acc-val_acc))
    ofs = sum(gap) / (n * (n-1) / 2)
    return ofs

def under_fitting_score(h):
    gap = []
    for i in h.epoch:
        acc = h.history['acc'][i]
        val_acc = h.history['val_accuracy'][i]
        gap.append(abs(acc-val_acc))
    gap = np.array(gap)
    return gap.mean()

def find_best_epoch(h, thresh=0.02):
    epochs = []
    for i in h.epoch:
        acc = h.history['acc'][i]
        val_acc = h.history['val_accuracy'][i]
        if abs(acc-val_acc) <= thresh:
            epochs.append(i)

    if not epochs:
        print ("No result")
        return None
    max_e = -1
    max_val_acc = -1
    for i in epochs:
        if h.history['val_accuracy'][i] > max_val_acc:
            max_e = i
            max_val_acc = h.history['val_acc'][i]
    print ("best epoch = %d ; best acc = %.6f ; best val_acc = %.6f" % (max_e, h.history['acc'][max_e], h.history['val_acc'][max_e]))
    return max_e

def success_rate(model, X_test, y_test):
    y_pred = model.predict_classes(X_test)
    n = len(y_pred)
    s = 0
    for i in range(n):
        if y_pred[i] == y_test[i]:
            s += 1
    return float(s) / n

def save_model_summary(model, filename):
    current_stdout = sys.stdout
    f = file(filename, 'w')
    sys.stdout = f
    model.summary()
    sys.stdout = current_stdout
    f.close()
    return filename