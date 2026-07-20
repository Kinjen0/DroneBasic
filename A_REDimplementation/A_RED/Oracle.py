"""# Oracle"""
from bidict import bidict
import numpy as np

class Oracle:
  def __init__(self, X, y):
    self.X = X #[[data]]
    self.y = y #[[label, relevant]]
    labels_only = np.array(y)[:,0]
    self.num_classes = len(np.unique(labels_only))
    self.int_str_label_bidict = self.create_label_bidict(labels_only)

  def create_label_bidict(self,labels):
      """
      Create a bidirectional dictionary mapping string labels to unique integers.

      Parameters:
      labels : list or array-like
          List of string labels (can contain duplicates).

      Returns:
      int_str_label_dict : bidict
          Bidirectional dictionary mapping strings to integers and vice versa.
      """
      # Convert input to numpy array and get unique labels
      unique_labels = np.unique(np.array(labels, dtype=str))

      # Create bidict with string -> integer and integer -> string mappings
      int_str_label_bidict = bidict({label: idx for idx, label in enumerate(unique_labels)})

      return int_str_label_bidict

  def answer_query(self, abs_index):
    if (abs_index >= self.X.shape[0]):
      print("Index out of range", abs_index, self.X.shape[0])
    label = self.y[abs_index][0]
    relevance = self.y[abs_index][1]
    return label, relevance