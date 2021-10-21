#!/usr/bin/env python
import torch
from typing import Tuple, List
from models.modules.emission_scorer_base import EmissionScorerBase


class FewShotTextClassifier(torch.nn.Module):
    def __init__(self,
                 opt,
                 emission_scorer: EmissionScorerBase,
                 decoder: torch.nn.Module,
                 config: dict = None,  # store necessary setting or none-torch params
                 emb_log: str = None):
        super(FewShotTextClassifier, self).__init__()
        self.opt = opt
        self.emission_scorer = emission_scorer
        self.decoder = decoder
        self.config = config
        self.emb_log = emb_log

        # record prototype data
        self.intent_prototype = None
        self.intent_emission = None

    def cal_prototype(self,
                      test_reps: torch.Tensor,
                      test_output_mask: torch.Tensor,
                      support_reps: torch.Tensor,
                      support_output_mask: torch.Tensor,
                      test_target: torch.Tensor,
                      support_target: torch.Tensor):
        test_output_mask = torch.ones(test_output_mask.shape[0], 1).to(test_output_mask.device)  # for sc, each test has only 1 output
        self.intent_prototype = self.emission_scorer.cal_prototype(test_reps, support_reps, test_output_mask,
                                                                   support_output_mask, support_target)

        return self.intent_prototype

    def update_prototype(self, prototype_reps):
        self.intent_prototype = self.emission_scorer.update_prototype(prototype_reps)

    def set_prototype(self, intent_prototype):
        self.emission_scorer.set_prototype(intent_prototype)

    def get_prototype(self):
        self.intent_prototype = self.emission_scorer.get_prototype()
        return self.intent_prototype

    def cal_emission(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            prototype_reps: torch.Tensor = None,
    ):
        # calculate emission: shape(batch_size, 1, no_pad_num_tag)
        test_output_mask = torch.ones(test_output_mask.shape[0], 1).to(test_output_mask.device)  # for sc, each test has only 1 output
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target)
        self.get_prototype()
        self.intent_emission = emission
        return emission

    def cat_emission(self, slot_emission):
        if slot_emission is not None:
            self.intent_emission += slot_emission

    def forward(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_mask: torch.Tensor = None,
            prototype_reps: torch.Tensor = None,
            slot_emission: torch.Tensor = None,
    ):
        """
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, multi-label_num)
        :param support_target: one-hot targets (batch_size, support_size, multi-label_num, num_tags)
        :param label_mask: the output label mask
        :param prototype_reps: the fix prototype
        :param slot_emission: the slot emission
        :return:
        """

        logits = self.intent_emission
        # as we remove pad label (id = 0), so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)
        loss = self.decoder.forward(logits=logits, mask=test_output_mask, tags=test_target)
        return loss

    def decode(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_mask: torch.Tensor = None,
            prototype_reps: torch.Tensor = None,
            slot_emission: torch.Tensor = None,
    ):
        """
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, multi-label_num)
        :param support_target: one-hot targets (batch_size, support_size, multi-label_num, num_tags)
        :param label_mask: the output label mask
        :param prototype_reps: the fix prototype
        :param slot_emission: the slot emission
        :return:
        """

        logits = self.intent_emission
        # as we remove pad label (id = 0), so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)

        prediction = self.decoder.decode(logits=logits)
        # we block pad label(id=0) before by - 1, here, we add 1 back
        prediction = self.add_back_pad_label(prediction)
        return prediction

    def add_back_pad_label(self, predictions: List[List[int]]):
        for pred in predictions:
            for ind, l_id in enumerate(pred):
                pred[ind] += 1  # pad token is in the first place
        return predictions

    def get_emission(self):
        return self.intent_emission


class SchemaFewShotTextClassifier(FewShotTextClassifier):
    def __init__(
            self,
            opt,
            emission_scorer: EmissionScorerBase,
            decoder: torch.nn.Module,
            config: dict = None,  # store necessary setting or none-torch params
            emb_log: str = None):
        super(SchemaFewShotTextClassifier, self).__init__(opt, emission_scorer, decoder, config, emb_log)

    def cal_prototype(self,
                      test_reps: torch.Tensor,
                      test_output_mask: torch.Tensor,
                      support_reps: torch.Tensor,
                      support_output_mask: torch.Tensor,
                      test_target: torch.Tensor,
                      support_target: torch.Tensor,
                      label_reps: torch.Tensor = None,):
        test_output_mask = torch.ones(test_output_mask.shape[0], 1).to(test_output_mask.device)  # for sc, each test has only 1 output
        self.intent_prototype = self.emission_scorer.cal_prototype(test_reps, support_reps, test_output_mask,
                                                                   support_output_mask, support_target, label_reps)
        return self.intent_prototype

    def cal_emission(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            prototype_reps: torch.Tensor = None,
            label_reps: torch.Tensor = None,
    ):
        # calculate emission: shape(batch_size, 1, no_pad_num_tag)
        test_output_mask = torch.ones(test_output_mask.shape[0], 1).to(test_output_mask.device)  # for sc, each test has only 1 output
        emission = self.emission_scorer(test_reps, support_reps, test_output_mask, support_output_mask, support_target,
                                        label_reps)
        self.get_prototype()
        self.intent_emission = emission
        return emission

    def forward(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_reps: torch.Tensor = None,
            label_mask: torch.Tensor = None,
            prototype_reps: torch.Tensor = None,
            slot_emission: torch.Tensor = None,
    ):
        """
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, multi-label_num)
        :param support_target: one-hot targets (batch_size, support_size, multi-label_num, num_tags)
        :param label_reps: the output label mask
        :param label_mask: the output label mask
        :param prototype_reps: the fix prototype
        :param slot_emission: the slot emission
        :return:
        """
        logits = self.intent_emission
        # as we remove pad label (id = 0), so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)
        loss = self.decoder.forward(logits=logits, mask=test_output_mask, tags=test_target)
        return loss

    def decode(
            self,
            test_reps: torch.Tensor,
            test_output_mask: torch.Tensor,
            support_reps: torch.Tensor,
            support_output_mask: torch.Tensor,
            test_target: torch.Tensor,
            support_target: torch.Tensor,
            label_reps: torch.Tensor = None,
            label_mask: torch.Tensor = None,
            prototype_reps: torch.Tensor = None,
            slot_emission: torch.Tensor = None,
    ):
        """
        :param test_reps: (batch_size, test_len, emb_dim)
        :param test_output_mask: (batch_size, test_len)
        :param support_reps: (batch_size, support_size, support_len, emb_dim)
        :param support_output_mask: (batch_size, support_size, support_len)
        :param test_target: index targets (batch_size, multi-label_num)
        :param support_target: one-hot targets (batch_size, support_size, multi-label_num, num_tags)
        :param label_reps: the output label mask
        :param label_mask: the output label mask
        :param prototype_reps: the fix prototype
        :param slot_emission: the slot emission
        :return:
        """
        logits = self.intent_emission
        # as we remove pad label (id = 0), so all label id sub 1. And relu is used to avoid -1 index
        test_target = torch.nn.functional.relu(test_target - 1)

        prediction = self.decoder.decode(logits=logits)
        # we block pad label(id=0) before by - 1, here, we add 1 back
        prediction = self.add_back_pad_label(prediction)
        return prediction


def main():
    pass


if __name__ == "__main__":
    main()
