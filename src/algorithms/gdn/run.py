import torch.nn.functional as F
import torch.nn as nn
import torch


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def train(model=None, save_path='', lr=0.1, decay=0, epoch=100, train_dataloader=None, val_dataloader=None, device='cuda'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    train_loss_list = []
    min_loss = 1e+8

    epoch = epoch
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader
    for i_epoch in range(epoch):
        acu_loss = 0
        # labels are real value, and the model generate predicted values
        for x, labels, _, edge_index in dataloader:
            # _start = src.GDN_AAAI21.util.time()
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
        # if (i_epoch+1) % 10 == 0 or (i_epoch+1) == 1:
        #     print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
        #                     i_epoch+1, epoch, acu_loss/len(dataloader), acu_loss), flush=True)

        # use val dataset to judge
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break
        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_list


def test(model, dataloader, device='cuda'):
    # test
    loss_func = nn.MSELoss(reduction='mean')

    test_loss_list = []
    # now = src.GDN_AAAI21.util.time()

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()
    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            loss = loss_func(predicted, y)
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        # if i % 10000 == 1 and i > 1:
        #     print(timeSincePlus(now, i / test_len))

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list]
    # return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]

