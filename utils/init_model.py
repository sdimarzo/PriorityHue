from utils.init_weights import init_weights

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model