using log4net;
using log4net.Config;
using log4net.Repository;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    internal class Class1
    {
        static void Main(string[] args)
        {

            ILoggerRepository repository = LogManager.CreateRepository("NETCoreRepository");
            BasicConfigurator.Configure(repository);
            XmlConfigurator.Configure(repository, new FileInfo("log4net.config"));
            ILog log = LogManager.GetLogger(repository.Name, "NETCorelog4net");

            log.Info("NETCorelog4net log");
            log.Error("use test menthod to definte the log4" +
                "");
            log.Error("error");
            log.Warn("warn");
            Console.ReadKey();

        }



    }
}
