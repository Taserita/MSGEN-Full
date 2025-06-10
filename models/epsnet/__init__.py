from .dualenc import DualEncoderEpsNetwork, ConditionalDualEncoderEpsNetwork

def get_model(config):
    if config.network == 'dualenc':
        return DualEncoderEpsNetwork(config)
    elif config.network == 'condition':
        return ConditionalDualEncoderEpsNetwork(config)
    else:
        raise NotImplementedError('Unknown network: %s' % config.network)
