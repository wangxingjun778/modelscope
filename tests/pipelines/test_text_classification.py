# Copyright (c) Alibaba, Inc. and its affiliates.
import tempfile
import unittest
import zipfile
from pathlib import Path

from maas_lib.fileio import File
from maas_lib.models import Model
from maas_lib.models.nlp import SequenceClassificationModel
from maas_lib.pipelines import SequenceClassificationPipeline, pipeline
from maas_lib.preprocessors import SequenceClassificationPreprocessor


class SequenceClassificationTest(unittest.TestCase):

    def predict(self, pipeline_ins: SequenceClassificationPipeline):
        from easynlp.appzoo import load_dataset

        set = load_dataset('glue', 'sst2')
        data = set['test']['sentence'][:3]

        results = pipeline_ins(data[0])
        print(results)
        results = pipeline_ins(data[1])
        print(results)

        print(data)

    def test_run(self):
        model_url = 'https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com' \
                    '/release/easynlp_modelzoo/alibaba-pai/bert-base-sst2.zip'
        cache_path_str = r'.cache/easynlp/bert-base-sst2.zip'
        cache_path = Path(cache_path_str)

        if not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.touch(exist_ok=True)
            with cache_path.open('wb') as ofile:
                ofile.write(File.read(model_url))

        with zipfile.ZipFile(cache_path_str, 'r') as zipf:
            zipf.extractall(cache_path.parent)
        path = r'.cache/easynlp/bert-base-sst2'
        model = SequenceClassificationModel(path)
        preprocessor = SequenceClassificationPreprocessor(
            path, first_sequence='sentence', second_sequence=None)
        pipeline1 = SequenceClassificationPipeline(model, preprocessor)
        self.predict(pipeline1)
        pipeline2 = pipeline(
            'text-classification', model=model, preprocessor=preprocessor)
        print(pipeline2('Hello world!'))

    def test_run_modelhub(self):
        model = Model.from_pretrained('damo/bert-base-sst2')
        preprocessor = SequenceClassificationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
        pipeline_ins = pipeline(
            task='text-classification', model=model, preprocessor=preprocessor)
        self.predict(pipeline_ins)


if __name__ == '__main__':
    unittest.main()