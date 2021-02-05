from two_class_learning import * 
import numpy as np 
import matplotlib.pyplot as plt
import time
import csv

def generalizationError(populationError, empiricalError):
    """
    Compute the generalization error as |populationError - empiricalError|
    """
    return abs(populationError - empiricalError)


def classificationError():
    """
    Function which computes the classification error over every model
    """
    print('Computing Classification Test Error')

    dl1, dl2 = CIFAR10TestLoader() 
    err1, err2 = [], []

    # load every model 
    with open('classification_error.csv','w') as clf:
        writer = csv.writer(clf)
        writer.writerow(['Classification Error Dataset 1','Classification Error Dataset 2'])

        for i in tqdm(range(5,95,2)):
            mtl = MTL()
            PATH = os.getcwd() + '/mtl_model_{d1}_vs_{d2}.pth'.format(d1=i, d2=100-i)
            mtl.load_state_dict(state_dict=torch.load(PATH))

            e1, e2 = accuracy(dl1, mtl), accuracy(dl2, mtl)
            err1.append(e1)
            err2.append(e2)
            writer.writerow([e1,e2])
    
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot([0.05 + 0.02* i for i in range(45)], err1, color='black')
    ax_right.plot([0.05 + 0.02* i for i in range(45)], err2, color='red')
    ax_right.set_ylabel('Dataset 2 (red) accuracy')
    ax_left.set_ylabel('Dataset 1 (black) accuracy')
    ax_left.set_xlabel('Percentage of points from dataset 1 in training set')
    plt.title('Classification Accuracy over both datasets')
    plt.savefig('classification_error_plots.pdf')


def main():
    # define proportions used for training
    rg = [0.05 + 0.02* i for i in range(45)]
    neg = [0.95 - 0.02* i for i in range(45)]
    genError1, genError2 = [], []

    # prep the csv file for data to be written to it
    with open('overfitting_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Generalization Error Dataset 1','Generalization Error Dataset 2'])

        for i in range(len(rg)):
            p1, p2 = rg[i], neg[i]
            print('p1 {p1}, p2 {p2}'.format(p1=p1, p2=p2))

            # compute training loss
            e1, e2 = train(dataOnePer=p1, dataTwoPer=p2)

            # compute approximate population loss
            ploss1, ploss2 = test(p1, p2)

            # compute generalization error
            g1, g2 = generalizationError(ploss1, e1), generalizationError(ploss2, e2)
            genError1.append(g1)
            genError2.append(g2)
            
            # write data to a csv to save the data
            writer.writerow([g1,g2])
    
    # plot the results and save the plot
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()
    ax_left.plot([0.05 + 0.02* i for i in range(45)], genError1, color='black')
    ax_right.plot([0.05 + 0.02* i for i in range(45)], genError2, color='red')
    ax_right.set_ylabel('Dataset 2 (red)')
    ax_left.set_ylabel('Dataset 1 (black)')
    ax_left.set_xlabel('Percentage of points from dataset 1 in training set')
    plt.title('Generalization Error over both datasets')
    plt.savefig('overfitting_plots.pdf')

if __name__ == "__main__":
    start_time = time.time()
    # main()
    classificationError()
    print("--- %s seconds ---" % (time.time() - start_time))