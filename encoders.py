class LabelEncoder:

  def __init__(self, labels: list[str]):
    """
        Inizializza l'encodera partire da una lista di etichette.

        :param labels: Una lista di stringhe, ciascuna rappresentante un'etichetta.
        """
    self.labels = {label: i + 1 for i, label in enumerate(labels)}

  def get_label(self, i: int) -> str:
    """
        Ottiene il nome dell'etichetta dato un indice.

        :param i: Un intero rappresentante l'indice dell'etichetta.
        :return: Una stringa che rappresenta l'etichetta corrispondente.
        """
    return list(self.labels.keys())[list(self.labels.values()).index(i)]

  def get_index(self, label: str) -> int:
    """
        Ottiene l'indice di un'etichetta dato il suo nome.

        :param label: Una stringa che rappresenta il nome dell'etichetta.
        :return: Un intero che rappresenta l'indice corrispondente all'etichetta.
        """
    return self.labels.get(label)


class LabelEncoderUNIMIB2016:

  def __init__(self, labels: list[str]):
    """
        Inizializza l'encodera partire da una lista di etichette.

        :param labels: Una lista di stringhe, ciascuna rappresentante un'etichetta.
        """
    self.labels = {label: i for i, label in enumerate(labels)}

  def get_label(self, i: int) -> str:
    """
        Ottiene il nome dell'etichetta dato un indice.

        :param i: Un intero rappresentante l'indice dell'etichetta.
        :return: Una stringa che rappresenta l'etichetta corrispondente.
        """
    return list(self.labels.keys())[list(self.labels.values()).index(i)]

  def get_index(self, label: str) -> int:
    """
        Ottiene l'indice di un'etichetta dato il suo nome.

        :param label: Una stringa che rappresenta il nome dell'etichetta.
        :return: Un intero che rappresenta l'indice corrispondente all'etichetta.
        """
    return self.labels.get(label)
