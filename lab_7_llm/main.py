"""
Neural machine translation module.
"""
# pylint: disable=too-few-public-methods, undefined-variable, too-many-arguments, super-init-not-called
from pathlib import Path
from typing import Iterable, Sequence

import torch
from datasets import load_dataset
from evaluate import load
from pandas import DataFrame, read_csv
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchinfo import torchinfo
from transformers import AlbertForSequenceClassification, AutoTokenizer

from core_utils.llm.llm_pipeline import AbstractLLMPipeline
from core_utils.llm.metrics import Metrics
from core_utils.llm.raw_data_importer import AbstractRawDataImporter
from core_utils.llm.raw_data_preprocessor import AbstractRawDataPreprocessor, ColumnNames
from core_utils.llm.task_evaluator import AbstractTaskEvaluator
from core_utils.llm.time_decorator import report_time


class RawDataImporter(AbstractRawDataImporter):
    """
    A class that imports the HuggingFace dataset.
    """

    @report_time
    def obtain(self) -> None:
        """
        Download a dataset.

        Raises:
            TypeError: In case of downloaded dataset is not pd.DataFrame
        """
        dataframe = load_dataset(self._hf_name, split="test").to_pandas()
        if not isinstance(dataframe, DataFrame):
            raise TypeError()
        self._raw_data = dataframe


class RawDataPreprocessor(AbstractRawDataPreprocessor):
    """
    A class that analyzes and preprocesses a dataset.
    """

    def analyze(self) -> dict:
        """
        Analyze a dataset.

        Returns:
            dict: Dataset key properties
        """
        return {
            "dataset_number_of_samples": self._raw_data.shape[0],
            "dataset_columns": self._raw_data.shape[1],
            "dataset_duplicates": self._raw_data.duplicated().sum(),
            "dataset_empty_rows": self._raw_data.isna().sum().sum(),
            "dataset_sample_min_len": len(min(self._raw_data["text"], key=len)),
            "dataset_sample_max_len": len(max(self._raw_data["text"], key=len))
        }

    @report_time
    def transform(self) -> None:
        """
        Apply preprocessing transformations to the raw dataset.
        """
        self._data = (
            self._raw_data.rename(
                columns={
                    "text": ColumnNames.SOURCE.value,
                    "label": ColumnNames.TARGET.value,
                }
            )
            # .sample() shuffles the dataset.
            # This is necessary to fix the f1-measure:
            # The imdb dataset is sorted by label,
            # Which means for a smaller sample f1 can only
            # Be equal to 0 or 1.
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        self._data.replace({True: 1, False: 0}, inplace=True)


class TaskDataset(Dataset):
    """
    A class that converts pd.DataFrame to Dataset and works with it.
    """

    def __init__(self, data: DataFrame) -> None:
        """
        Initialize an instance of TaskDataset.

        Args:
            data (pandas.DataFrame): Original data
        """
        self._data = data

    def __len__(self) -> int:
        """
        Return the number of items in the dataset.

        Returns:
            int: The number of items in the dataset
        """
        return len(self._data)

    def __getitem__(self, index: int) -> tuple[str, ...]:
        """
        Retrieve an item from the dataset by index.

        Args:
            index (int): Index of sample in dataset

        Returns:
            tuple[str, ...]: The item to be received
        """
        return (self._data.iloc[index][ColumnNames.SOURCE.value],)

    @property
    def data(self) -> DataFrame:
        """
        Property with access to preprocessed DataFrame.

        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        return self._data


class LLMPipeline(AbstractLLMPipeline):
    """
    A class that initializes a model, analyzes its properties and infers it.
    """
    _model: torch.nn.Module

    def __init__(
            self,
            model_name: str,
            dataset: TaskDataset,
            max_length: int,
            batch_size: int,
            device: str
    ) -> None:
        """
        Initialize an instance of LLMPipeline.

        Args:
            model_name (str): The name of the pre-trained model
            dataset (TaskDataset): The dataset used
            max_length (int): The maximum length of generated sequence
            batch_size (int): The size of the batch inside DataLoader
            device (str): The device for inference
        """
        super().__init__(model_name, dataset, max_length, batch_size, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AlbertForSequenceClassification.from_pretrained(
            self._model_name, num_labels=2)

    def analyze_model(self) -> dict:
        """
        Analyze model computing properties.

        Returns:
            dict: Properties of a model
        """

        embeddings_length = self._model.config.max_position_embeddings
        ids = torch.ones(1, embeddings_length, dtype=torch.long)

        data = {
            'input_ids': ids,
            'attention_mask': ids
        }

        model_summary = torchinfo.summary(
            self._model,
            input_data=data,
            verbose=0,
        )

        return {
            "input_shape": {
                "attention_mask": list(model_summary.input_size["attention_mask"]),
                "input_ids": list(model_summary.input_size["input_ids"])
            },
            "embedding_size": embeddings_length,
            "output_shape": model_summary.summary_list[-1].output_size,
            "num_trainable_params": model_summary.trainable_params,
            "vocab_size": self._model.config.vocab_size,
            "size": model_summary.total_param_bytes,
            "max_context_length": self._model.config.max_length
        }

    @report_time
    def infer_sample(self, sample: tuple[str, ...]) -> str | None:
        """
        Infer model on a single sample.

        Args:
            sample (tuple[str, ...]): The given sample for inference with model

        Returns:
            str | None: A prediction
        """
        if self._model is None:
            return None
        return self._infer_batch([sample])[0]

    @report_time
    def infer_dataset(self) -> DataFrame:
        """
        Infer model on a whole dataset.

        Returns:
            pd.DataFrame: Data with predictions
        """
        loader = DataLoader(self._dataset, batch_size=self._batch_size)
        prediction = []

        for batch in loader:
            prediction.extend(self._infer_batch(batch))

        print("Predictions:", prediction)
        print("Targets:    ",
              self._dataset.data[ColumnNames.TARGET.value].tolist())

        self._dataset.data["predictions"] = prediction
        return DataFrame({
            "target": self._dataset.data[ColumnNames.TARGET.value].tolist(),
            "predictions": prediction
        })

    @torch.no_grad()
    def _infer_batch(self, sample_batch: Sequence[tuple[str, ...]]) -> list[str]:
        """
        Infer model on a single batch.

        Args:
            sample_batch (Sequence[tuple[str, ...]]): Batch to infer the model

        Returns:
            list[str]: Model predictions as strings
        """
        if self._model is None:
            return []

        inputs = self._tokenizer(
            sample_batch[0],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length
        ).to(self._device)

        outputs = self._model(**inputs)

        return list(str(prediction.item()) for prediction in torch.argmax(outputs.logits, dim=1))


class TaskEvaluator(AbstractTaskEvaluator):
    """
    A class that compares prediction quality using the specified metric.
    """

    def __init__(self, data_path: Path, metrics: Iterable[Metrics]) -> None:
        """
        Initialize an instance of Evaluator.

        Args:
            data_path (pathlib.Path): Path to predictions
            metrics (Iterable[Metrics]): List of metrics to check
        """
        super().__init__(metrics)
        self._data_path = data_path

    @report_time
    def run(self) -> dict | None:
        """
        Evaluate the predictions against the references using the specified metric.

        Returns:
            dict | None: A dictionary containing information about the calculated metric
        """
        predictions_df = read_csv(self._data_path)

        evaluations = {}
        for metric in self._metrics:
            metric_instance = load(metric.value)
            evaluations[metric.value] = metric_instance.compute(
                predictions=predictions_df["predictions"].tolist(),
                references=predictions_df["target"].tolist(),
            )[metric.value]
        return evaluations
