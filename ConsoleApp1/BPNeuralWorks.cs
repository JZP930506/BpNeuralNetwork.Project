
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Path = System.IO.Path;

namespace ConsoleApp1
{
	internal class BPNeuralWorks
	{
		private static int IM = 4; // 输入层数量
		private static int OM = 1; // 输出层数量
		private static int RM = 8; // 隐含层数量
								   //private static  Path path = Path.get("E:/Users/fanxin/Workspaces/MyEclipse 2017 CI/BP newwork/src/Iris.txt");
		private double learnRate = 0.5; // 学习速率
		private double[] thresholdHide = new double[RM];
		private double[] thresholdOut = new double[OM];
		private List<List<Double>> trainData = new List<List<Double>>();
		private List<List<int>> trainLabel = new List<List<int>>();
		private double[][] nom_data; // 归一化输入数据中的最大值和最小值
		private double[][] Win = new double[IM][]; // 输入到隐含连接权值
		private double[][] Wout = new double[RM][]; // 隐含到输出连接权值
		private double[] Ek = new double[OM];
		private double J = 0.1;
		double[][] out1 = new double[3][];

		public BPNeuralWorks()
		{
			// 初始化权值和清零//
			//readData(path);
			//NormalizeData(trainData);
			//for (int i = 0; i < trainData.Count(); i++)
			//{
			//	for (int j = 0; j < trainData.get(i).size(); j++)
			//	{
			//		trainData.get(i).set(j, Normalize(trainData.get(i).get(j), nom_data[j][0], nom_data[j][1]));
			//	}
			//}
			//InitNetWork();
		}

		// 初始化网络的权值和阈值
		public void InitNetWork()
		{
			// 初始化上一次权值量,范围为-0.5-0.5之间
			// in_hd_last = new double[IM][RM];
			// hd_out_last = new double[RM][OM];

			for (int i = 0; i < IM; i++)
				for (int j = 0; j < RM; j++)
				{
					int flag = 1; // 符号标志位(-1或者1)
					if ((new Random().NextInt64(2)) == 1)
						flag = 1;
					else
						flag = -1;
					Win[i][j] = (new Random().NextDouble() / 2) * flag; // 初始化in-hidden的权值
				}

			for (int i = 0; i < RM; i++)
				for (int j = 0; j < OM; j++)
				{
					int flag = 1; // 符号标志位(-1或者1)
					if ((new Random().NextInt64(2)) == 1)
						flag = 1;
					else
						flag = -1;
					Wout[i][j] = (new Random().NextDouble() / 2) * flag; // 初始化hidden-out的权值
				}

			// 阈值均初始化为0
			for (int k = 0; k < RM; k++)
				thresholdHide[k] = 0;

			for (int k = 0; k < OM; k++)
				thresholdOut[k] = 0;

		}

		public void train()
		{
			out1 = new double[3][];
			for (int iter = 0; iter < 2000; iter++)
			{
				for (int cnd = 0; cnd < trainData.Count(); cnd++)
				{
					// 第一层输入节点赋值
					for (int i = 0; i < IM; i++)
					{
						out1[0][i] = trainData[cnd][i]; // 为输入层节点赋值，其输入与输出相同
					}
					bpNetForwardProcess(trainLabel[cnd]); // 前向传播
					bpNetReturnProcess(); // 误差反向传播
										  // System.out.println("This is the " + (iter + 1) + " th
										  // trainning NetWork !");
										  // System.out.println("All Samples Accuracy is " + J);
				}
			}
		}

		public void test()
		{
			double count = 0;
			for (int cnd = 0; cnd < trainData.Count(); cnd++)
			{
				// 第一层输入节点赋值

				for (int i = 0; i < IM; i++)
				{
					out1[0][i] = trainData[cnd][i]; // 为输入层节点赋值，其输入与输出相同
				}
				count += predict(trainLabel[cnd]); // 前向传播
			}

		}
		public int predict(List<int> label)
		{
			bpNetForwardProcess(label); // 前向传播// 输出层S激活输出//
			bool flag = true;
			for (int j = 0; j < OM; j++)
			{
				if ((out1[2][j] > 0.5 && label[j] == 0) || (out1[2][j] < 0.5 && label[j] == 1)) {
					flag = false;
					break;
				}
			}
			if (flag) {
				return 1;
			}
			return 0;
		}

		public void bpNetForwardProcess(List<int> label)
		{
			// 隐含层权值和计算//
			// 计算隐层节点的输出值
			for (int j = 0; j < RM; j++)
			{
				double v = -thresholdHide[j];
				for (int i = 0; i < IM; i++)
					v += Win[i][j] * out1[0][i];
				out1[1][j] = 1 / (1 + Math.Exp(-v));
			}
			// 计算输出层节点的输出值
			for (int j = 0; j < OM; j++)
			{
				double v = -thresholdOut[j];
				for (int i = 0; i < RM; i++)
					v += Wout[i][j] * out1[1][i];
				out1[2][j] = 1 / (1 + Math.Exp(-v));
			}
			// 计算输出与理想输出的偏差//
			for (int k = 0; k < OM; k++)
			{
				Ek[k] = out1[2][k] - label[k];
			}
			J = 0;
			for (int k = 0; k < OM; k++)
			{
				J = J + Ek[k] * Ek[k] / 2.0;
			}

		}

		public void bpNetReturnProcess()
		{
			// 隐层到输出权值修正
			double[] g = new double[OM];
			for (int j = 0; j < OM; j++)
			{
				g[j] = Ek[j] * out1[2][j] * (1 - out1[2][j]);
			}
			for (int i = 0; i < RM; i++) {
				for (int j = 0; j < OM; j++) {
					Wout[i][j] += -learnRate * g[j] * out1[1][i]; // 未加权值动量项
				}

			}
			double[] e = new double[RM];
			// 计算隐层的delta值
			for (int h = 0; h < RM; h++)
			{
				double t = 0;
				for (int j = 0; j < OM; j++)
					t += Wout[h][j] * g[j];
				e[h] = t * out1[1][h] * (1 - out1[1][h]);
			}

			// 输入层和隐含层之间权值和阀值调整
			for (int i = 0; i < IM; i++)
			{
				for (int h = 0; h < RM; h++)
				{
					Win[i][h] += -learnRate * e[h] * out1[0][i]; // 未加权值动量项
				}
			}

			for (int j = 0; j < OM; j++)
				thresholdOut[j] += learnRate * g[j];
			for (int h = 0; h < RM; h++)
				thresholdHide[h] += learnRate * e[h];
		}


		public void ReadData(string path)
		{
			try {
				string fileContent = System.IO.File.ReadAllText(path);
				if (string.IsNullOrWhiteSpace(fileContent) == false)
				{
					var rowContent = fileContent.Split(Environment.NewLine).Where(p => string.IsNullOrEmpty(p) == false);
					foreach (var recod in rowContent.Skip(1))
					{
						var content = recod.Split(",");
						List<Double> tempList = new List<Double>();
						List<int> labelList = new List<int>();
						for (int i = 0; i < content.Count(); i++)
						{
							tempList.Add(Convert.ToDouble(content[i]));
						}
						trainData.Add(tempList);
						trainLabel.Add(labelList);
					}
				}
			} catch (Exception e)
			{
				throw new Exception(e.ToString());
			}
		}

		// 学习样本归一化,找到输入样本数据的最大值和最小值
		public void NormalizeData(List<List<Double>> trainData)
		{
			// 提前获得输入数据的个数
			int flag = 1;
			nom_data = new double[IM][];
			foreach (List<Double> list in trainData) {
				for (int i = 0; i < list.Count(); i++)
				{
					if (flag == 1)
					{
						nom_data[i][0] = Convert.ToDouble(list[i]);
						nom_data[i][1] = Convert.ToDouble(list[i]);
					}
					else
					{
						if (Convert.ToDouble(list[i]) > nom_data[i][0])
							nom_data[i][0] = Convert.ToDouble(list[i]);
						if (Convert.ToDouble(list[i]) < nom_data[i][1])
							nom_data[i][1] = Convert.ToDouble(list[i]);
					}
				}
				flag = 0;
			}
			/*
			 * for(int i=0;i<4;i++){ for(int j=0;j<2;j++){
			 * System.out.print(nom_data[i][j]+" "); } System.out.println(); }
			 */
		}

		// 归一化
		public double Normalize(double x, double max, double min)
		{
			double y = 0.1 + 0.8 * (x - min) / (max - min);
			return y;
		}
	}


}
