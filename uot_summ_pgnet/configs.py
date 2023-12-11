# -*- coding: utf-8 -*-
#pylint: skip-file
import os 

class CommonConfigs(object):
    def __init__(self, d_type):
        self.ROOT_PATH = os.getcwd() + "/"
        self.TRAINING_DATA_PATH = self.ROOT_PATH + d_type + "/train_set/"
        self.VALIDATE_DATA_PATH = self.ROOT_PATH + d_type + "/validate_set/"
        self.TESTING_DATA_PATH = self.ROOT_PATH + d_type + "/test_set/"
        self.RESULT_PATH = self.ROOT_PATH + d_type + "/result/"
        self.MODEL_PATH = self.ROOT_PATH + d_type + "/model/"
        self.BEAM_SUMM_PATH = self.RESULT_PATH + "/beam_summary/"
        self.BEAM_GT_PATH = self.RESULT_PATH + "/beam_ground_truth/"
        self.GROUND_TRUTH_PATH = self.RESULT_PATH + "/ground_truth/"
        self.SUMM_PATH = self.RESULT_PATH + "/summary/"
        self.TMP_PATH = self.ROOT_PATH + d_type + "/tmp/"


class DeepmindTraining(object):
    IS_UNICODE = False
    REMOVES_PUNCTION = False
    HAS_Y = True
    BATCH_SIZE = 9
    ######## todo change 4 ot  ########
    # FIRST_STAGE_EPOCH_NUMS = 3
    # BATCH_SIZE_FIRST_STAGE = 9
    # BATCH_SIZE_SECOND_STAGE = 12


class DeepmindTesting(object):
    IS_UNICODE = False
    HAS_Y = True
    BATCH_SIZE = 5
    MIN_LEN_PREDICT = 35
    MAX_LEN_PREDICT = 120
    MAX_BYTE_PREDICT = None
    PRINT_SIZE = 50
    REMOVES_PUNCTION = False

    ######## todo change 4 ot  ########
    # https://arxiv.org/pdf/2004.06190.pdf
    MIN_LEN_PREDICT_ONE_SEC = 10
    MAX_LEN_PREDICT_ONE_SEC = 200

    MAX_LEN_PREDICT_TOTAL = 200


class DeepmindConfigs():
    
    cc = CommonConfigs("deepmind")
   
    CELL = "lstm"  # gru or lstm
    CUDA = True
    COPY = True
    COVERAGE = True
    BI_RNN = True
    BEAM_SEARCH = True
    BEAM_SIZE = 4
    AVG_NLL = True
    NORM_CLIP = 2
    if not AVG_NLL:
        NORM_CLIP = 5
    LR = 0.15

    DIM_X = 128
    DIM_Y = DIM_X

    MIN_LEN_X = 10
    MIN_LEN_Y = 10
    MAX_LEN_X = 400
    MAX_LEN_Y = 100
    MIN_NUM_X = 1
    MAX_NUM_X = 1
    MAX_NUM_Y = None

    NUM_Y = 1
    HIDDEN_SIZE = 256

    UNI_LOW_FREQ_THRESHOLD = 10

    PG_DICT_SIZE = 50000  # dict for acl17 paper: pointer-generator
    
    W_UNK = "<unk>"
    W_BOS = "<bos>"
    W_EOS = "<eos>"
    W_PAD = "<pad>"
    W_LS = "<s>"
    W_RS = "</s>"

    ######## 4 ot model########
    LR4MATCHER = 1e-5

    # CHANGE_MODE_EPOCH_NUM = 0  # after certain epochs, change the mode

    BI_RNN_SECTION_EXT = True

    DIM_SECTION_EXT = HIDDEN_SIZE

    MAX_LEN_ONE_SECTION_SRC = 600  # 500 4 pubmed; 600 4 arxiv
    MAX_LEN_ONE_SENTENCE_ABST = 40
    RECOMBINED_MAX_LEN_ABST = 140  # 80 4 pubmed; 120 4 arxiv

    MAX_SRC_SECTIONS_NUMS = 20
    MAX_ABST_SENTENCES_NUMS = 20

    HIDDEN_SIZE_OT_EXTRACTOR = 128

    EPSILON = 0.006
    TAU_SINKHORN = 0.03

    # TAU_NN = 5.0

    BUCKET_SIZE_4_AVERAGE = 80
    # KESI = 1.0
    # MAX_LEN_ONE_SEN_Y = 50
    # MIN_LEN_ONE_SEN_Y = 0
    # MIN_LEN_ONE_SEN = 0



