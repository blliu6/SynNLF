import os
import csv
from utils.Config_B import CegisConfig
from benchmarks.Exampler_B import Example
from datetime import datetime


class SaveResult:
    def __init__(self, config: CegisConfig, runtime, B):
        self.path = '../benchmarks/result/'
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.config = config
        self.runtime = runtime
        self.B = str(B)

    def save_txt(self):
        file_name = self.config.EXAMPLE.name + '.txt'
        data = ["ACTIVATION", "N_HIDDEN_NEURONS", "MULTIPLICATOR", "BATCH_SIZE", "LEARNING_RATE",
                "LOSS_WEIGHT", "MARGIN", "SPLIT_D", "DEG", "R_b"]
        with open(self.path + file_name, 'a', encoding='utf-8') as f:
            f.write(f"time of recording: {datetime.now()}")
            f.write("The configuration information is as follows:\n")
            for key in data:
                f.write(f'   {key}: {self.config.__getattribute__(key)}\n')
                # f.write('   ' + key + ':' + str(self.config.__getattribute__(key)) + '\n')
            f.write('\nThe runtime is as follows:\n')
            f.write('   Total learning time: {}\n'.format(self.runtime[0]))
            f.write('   Total counter-examples generating time: {}\n'.format(self.runtime[1]))
            f.write('   Total sos verifying time: {}\n'.format(format(self.runtime[2])))
            f.write('\nThe synthetic Barrier Certificate is as follows:\n')
            f.write(f'   Iter: {self.runtime[3]}\n')
            f.write('   B=')
            for i in range(0, len(self.B)):
                f.write(self.B[i])
                if (i + 6) % 150 == 0 and i != 0:
                    f.write('\n')
            f.write('\n\n')

    def save_csv(self):
        header = ['time of recording', 'name', 'n', 'nhl', 'nm', 't_learn', 't_cex', 't_sos', 'rate', 'batch_size',
                  'iter', 'activation']
        if not os.path.exists(self.path + 'result.csv'):
            os.mknod(self.path + 'result.csv')
            f = open(self.path + 'result.csv', 'w', encoding='utf-8')
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

        with open(self.path + 'result.csv', 'a', encoding='utf-8') as f:
            data = [datetime.now(), self.config.EXAMPLE.name, self.config.EXAMPLE.n, self.config.N_HIDDEN_NEURONS[0],
                    self.config.MULTIPLICATOR_NET, self.runtime[0], self.runtime[1], self.runtime[2],
                    self.config.LEARNING_RATE, self.config.BATCH_SIZE, self.runtime[3], self.config.ACTIVATION]
            writer = csv.writer(f)
            writer.writerow(data)


if __name__ == '__main__':
    '''
    test code
    '''
    example = Example(
        n=7,
        D_zones=[[-2, 2]] * 7,
        I_zones=[[-1.01, -0.99]] * 7,
        U_zones=[[1.8, 2]] * 7,
        f=[
            lambda x: -0.4 * x[0] + 5 * x[2] * x[3],
            lambda x: 0.4 * x[0] - x[1],
            lambda x: x[1] - 5 * x[2] * x[3],
            lambda x: 5 * x[4] * x[5] - 5 * x[2] * x[3],
            lambda x: -5 * x[4] * x[5] + 5 * x[2] * x[3],
            lambda x: 0.5 * x[6] - 5 * x[4] * x[5],
            lambda x: -0.5 * x[6] + 5 * x[4] * x[5]
        ],
        name='test_7dim'
    )
    opts = {
        "EXAMPLE": example
    }
    con = CegisConfig(**opts)
    B = "-1.98214774297e-10-1.25924660981e-12*x2-9.74452688236e-12*x5-9.74357368155e-12*x7+1.1449887559e-13*x10-6" \
        ".8936332256e-11*x2^2-1.46311839551e-09*x5^2-1.46311380156e-09*x7^2-2.15799504357e-09*x10^2-3.19654158345e-10" \
        "*x2*x5-3.19629800246e-10*x2*x7-2.56048044107e-09*x5*x7-2.00084051036e-12*x2*x10-1.07416966319e-10*x5*x10-1" \
        ".0742096333e-10*x7*x10"
    s = SaveResult(con, [1, 1, 1], B)
    s.save_csv()
