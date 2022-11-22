import torch
import tempfile,tarfile
import os
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.modeling import PRETRAINED_MODEL_ARCHIVE_MAP,CONFIG_NAME,cached_path

def get_config(config_path_or_type, logger):
    if config_path_or_type in PRETRAINED_MODEL_ARCHIVE_MAP:
        archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[config_path_or_type]
    else:
        archive_file = config_path_or_type
    # redirect to the cache, if necessary
    try:
        resolved_archive_file = cached_path(archive_file)
    except EnvironmentError:
        logger.error(
            "Model name '{}' was not found in model name list ({}). "
            "We assumed '{}' was a path or url but couldn't find any file "
            "associated to this path or url.".format(
                config_path_or_type,
                ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                archive_file))
        return None
    if resolved_archive_file == archive_file:
        logger.info("loading archive file {}".format(archive_file))
    else:
        logger.info("loading archive file {} from cache at {}".format(
            archive_file, resolved_archive_file))
    if os.path.isdir(resolved_archive_file):
        serialization_dir = resolved_archive_file
    else:
        # Extract archive to temp dir
        tempdir = tempfile.mkdtemp()
        logger.info("extracting archive file {} to temp dir {}".format(
            resolved_archive_file, tempdir))
        with tarfile.open(resolved_archive_file, 'r:gz') as archive:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, tempdir)
        serialization_dir = tempdir
    # Load config
    config_file = os.path.join(serialization_dir, CONFIG_NAME)
    config = BertConfig.from_json_file(config_file)
    return config

def valid_first(predict_mask, labels=None, logits=None):
    '''
    When not predicting the latter token(padding tokens),
    bring first the main token of logits, predict_mask, labels.
    this method is used when we don't want to add the loss of 'X' tokens

    :param logits: (batch_size, seq_len, num_labels)
    :param predict_mask: (batch_size, seq_len)
    :param labels: (batch_size, seq_len)
    :return: logits, predict_mask, labels in same size
    '''
    max_len = predict_mask.size(1)
    lens = predict_mask.sum(-1).cpu().numpy()# (batch_size)
    if labels is not None:
        flattened_labels = labels[predict_mask==1] # (sum_words_size)
    if logits is not None:
        flattened_logits = logits[predict_mask==1] # (sum_words_size, num_labels)
    flattened_masks = predict_mask[predict_mask==1] # (sum_words_size)

    def padded(v, max_len):
        if max_len-v.size()[0] > 0:
            pad = v.new_zeros([max_len-v.size()[0], *v.size()[1:]], requires_grad=False)
            return torch.cat([v, pad], 0)
        return v

    start_len = 0
    if logits is not None:
        new_logits = logits.clone()
    else:
        new_logits = None
    if labels is not None:
        new_labels = labels.clone()
    else:
        new_labels = None
    new_masks = predict_mask.clone()
    for ix, len in enumerate(lens):
        if logits is not None:
            new_logits[ix] = padded(flattened_logits[start_len:start_len + len, :], max_len)
        if labels is not None:
            new_labels[ix] = padded(flattened_labels[start_len:start_len + len], max_len)
        new_masks[ix] = padded(flattened_masks[start_len:start_len + len], max_len)
        start_len += len
    return new_masks, new_labels, new_logits

# logits = torch.tensor([[[1.2,2.1], [2.8,2.1], [2.2,-2.1]], [[4.1,2.2], [2.8,2.1], [2.2,-2.1]]]) # 2, 3, 2
# predict_mask = torch.tensor([[1,1,1],[1,0,1]]) # 2, 3
# labels = torch.tensor([[1,0,0],[0, 1, 1]]) # 2, 3
# print(valid_first(logits, predict_mask, labels))