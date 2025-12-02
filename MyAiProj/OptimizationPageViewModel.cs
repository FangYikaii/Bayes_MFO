using CommunityToolkit.Mvvm.Input;
using DbOperationLibrary;
using DocumentFormat.OpenXml.Drawing;
using DocumentFormat.OpenXml.Drawing.Diagrams;
using DocumentFormat.OpenXml.EMMA;
using DocumentFormat.OpenXml.Spreadsheet;
using DocumentFormat.OpenXml.Wordprocessing;
using DynamicData;
using MetalizationSystem.DataCollection;
using MetalizationSystem.DataServer;
using MetalizationSystem.Devices;
using MetalizationSystem.Domain;
using MetalizationSystem.EnumCollection;
using MetalizationSystem.ViewModels.Node;
using MetalizationSystem.Views;
using MetalizationSystem.Views.UC;
using Newtonsoft.Json;
using SqlSugar.Extensions;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Reflection;
using System.Threading;
using static MetalizationSystem.ViewModels.OptimizationPageViewModel;

namespace MetalizationSystem.ViewModels;

public static class StatusExtensions
{
    public static string GetDescription(this Status status)
    {
        return status switch
        {
            Status.AddAlgoProj => "add algorithm project",
            Status.InitAlgoProj => "init algorithm project",
            Status.UnSelectAlgoProj => "unselect algorithm project",
            Status.DeleteAlgoProj => "delete algorithm project",
            Status.RefreshAlgoProj => "refresh algorithm project",
            Status.NextBatchIter => "next batch iteration",
            Status.PrevBatchIter => "previous batch iteration",
            Status.DownloadAlgorithm => "download algorithm parameters",
            Status.UploadAlgothrim => "upload algorithm results",
            Status.ConfigBarCodeSuccess => "configure barcode successfully",
            Status.SubmitAdhensionValue => "submit adhesion value",

            _ => "Unknown"
        };
    }
}

public partial class OptimizationPageViewModel : ObservableObject
{

    #region data declaration

    private CommonDbOperation mOperation = null;
    private bool flagThrowSelectionChangedTrigger;
    private int CurrentDisplaytIterId = 0;
    public BindingList<BayesExperData> BayesExperDataList { get; set; }
    public BindingList<AlgoProjInfo> ExistingProjList { get; set; }

    [ObservableProperty]
    int _tabCtrlSelectedIndex;
    [ObservableProperty]
    string _newProjName;
    [ObservableProperty]
    string _newProjIterNum;
    [ObservableProperty]
    string _newProjSavePath;
    [ObservableProperty]
    string _newProjBatchNum;
    [ObservableProperty]
    string _existingProjName;
    [ObservableProperty]
    string _existingProjCreateTime;
    [ObservableProperty]
    AlgoProjInfo _existingSelectedProj;
    [ObservableProperty]
    string _selectedProjName;
    [ObservableProperty]
    string _selectedProjSavePath;
    [ObservableProperty]
    string _selectedProjIterNum;
    [ObservableProperty]
    string _selectedProjIterId;
    [ObservableProperty]
    string _selectedProjDisplayId;
    [ObservableProperty]
    string _selectedProjUploadId;
    [ObservableProperty]
    string _selectedProjDownloadId;
    [ObservableProperty]
    string _selectedProjAlgoGenId;
    [ObservableProperty]
    string _selectedProjAlgoRecevId;
    [ObservableProperty]
    string _selectedProjBarCodeId;
    [ObservableProperty]
    string _selectedProjExpId;
    [ObservableProperty]
    string _selectedProjBatchNum;
    [ObservableProperty]
    string _selectedProjCreateTime;
    [ObservableProperty]
    string _lblIterIdAndIterNum = "IterId/IterNum: None";
    [ObservableProperty]
    string _lblIterBatch = "IterBatch: None";
    [ObservableProperty]
    string _lblExpIdAndExpNum = "ExpId/ExpRange: None";
    [ObservableProperty]
    string _lblStatus = "Status: None";
    [ObservableProperty]
    string _lblDataVisualization = "Data Visualization";
    [ObservableProperty]
    string _bezierCurveImagePath;
    [ObservableProperty]
    string _umapImagePath;
    [ObservableProperty]
    string _paretoImagePath;
    [ObservableProperty]
    string _iterValueImagePath;
    [ObservableProperty]
    bool _iterBatchNextIsEnable = true;
    [ObservableProperty]
    bool _iterBatchPrevIsEnable = true;
    [ObservableProperty]
    bool _downloadParameterIsEnable = true;
    [ObservableProperty]
    bool _uploadParameterIsEnable = true;
    [ObservableProperty]
    bool _configBarCodeIsEnable = true;
    [ObservableProperty]
    BayesExperData _bayesSelectedItem;
    [ObservableProperty]
    Visibility _dataPlotVisibility = Visibility.Hidden;
    [ObservableProperty]
    int _bezierComboxSelectedIndex;
    [ObservableProperty]
    int _umapComboxSelectedIndex;

    public enum Status
    {
        AddAlgoProj,
        InitAlgoProj,
        UnSelectAlgoProj,
        DeleteAlgoProj,
        RefreshAlgoProj,
        NextBatchIter,
        PrevBatchIter,
        DownloadAlgorithm,
        UploadAlgothrim,
        ConfigBarCodeSuccess,
        SubmitAdhensionValue
    }


    #endregion

    public OptimizationPageViewModel()
    {
        mOperation = new CommonDbOperation(new CommonDbConnectionInfo()
        {
            DbType = SqlSugar.DbType.Sqlite,
            DbPath = Globa.Path.FileAlgoDB,
            DbPassword = ""
        });
        mOperation.InitTables(typeof(AlgoProjInfo), typeof(BayesExperData), typeof(FormulaData));
        BayesExperDataList = new BindingList<BayesExperData>();
        ExistingProjList = new BindingList<AlgoProjInfo>();

        BthEnableRefresh();
    }


    private void TestData()
    {
        Random random = new Random();
        int k = 1;
        for (int j = 1; j < 6; j++)
        {
            for (int i = 0; i < 5; i++)
            {
                var data = new BayesExperData()
                {
                    ProjName = ExistingSelectedProj.ProjName,
                    IterId = j,
                    ExpID = k,
                    Formula = random.Next(0, 100),
                    Concentration = random.Next(0, 100),
                    Temperature = random.Next(0, 100),
                    SoakTime = random.Next(0, 100),
                    PH = random.Next(0, 10)
                };
                mOperation.AddInfo(data);
                k++;
            }
        }
    }

    /// <summary>
    /// Button Enable Refresh
    /// </summary>
    private void BthEnableRefresh()
    {
        if (ExistingSelectedProj == null)
        {
            IterBatchNextIsEnable = false;
            IterBatchPrevIsEnable = false;
            DownloadParameterIsEnable = false;
            UploadParameterIsEnable = false;
            return;
        }
        if (CurrentDisplaytIterId == 0)
        {
            IterBatchPrevIsEnable = false;
            if (ExistingSelectedProj.DownloadId > 0)
            {
                IterBatchNextIsEnable = true;
            }
            else
            {
                IterBatchNextIsEnable = false;
            }
        }
        else
        {
            if (CurrentDisplaytIterId == 1)
            {
                IterBatchPrevIsEnable = false;
            }
            else
            {
                IterBatchPrevIsEnable = true;
            }
            if (CurrentDisplaytIterId < ExistingSelectedProj.DownloadId)
            {
                IterBatchNextIsEnable = true;
            }
            else
            {
                IterBatchNextIsEnable = false;
            }
        }

        if (ExistingSelectedProj.DownloadId < ExistingSelectedProj.IterId)
        {
            DownloadParameterIsEnable = true;
            UploadParameterIsEnable = false;
        }
        else
        {
            if (ExistingSelectedProj.UploadId < ExistingSelectedProj.IterId)
            {
                DownloadParameterIsEnable = false;
                UploadParameterIsEnable = true;
            }
            else
            {
                DownloadParameterIsEnable = true;
                UploadParameterIsEnable = false;
            }
        }

        if (ExistingSelectedProj.ConfigBarCodeId < ExistingSelectedProj.DownloadId)
        {
            ConfigBarCodeIsEnable = true;
        }
        else
        {
            ConfigBarCodeIsEnable = false;
        }
    }

    #region Project Command

    /// <summary>
    /// Proj TabCtrl Selection Changed Event
    /// </summary>
    [RelayCommand]
    public void TabCtrlSelectChanged()
    {

        if (TabCtrlSelectedIndex == 1 && !flagThrowSelectionChangedTrigger)
        {
            ExistingProjList.Clear();
            List<AlgoProjInfo> list = mOperation.GetInfo<AlgoProjInfo>();
            foreach (var item in list)
            {
                ExistingProjList.Add(item);
            }
        }
        flagThrowSelectionChangedTrigger = false;
    }

    /// <summary>
    /// Project DataGrid Selection Changed Event
    /// </summary>
    [RelayCommand]
    public void ProjDgSelectChanged()
    {
        flagThrowSelectionChangedTrigger = true;
    }

    /// <summary>
    /// Project List Refresh Command
    /// </summary>
    [RelayCommand]
    public void RefreshProj()
    {
        ExistingProjList.Clear();
        List<AlgoProjInfo> list = mOperation.GetInfo<AlgoProjInfo>();
        foreach (var item in list)
        {
            ExistingProjList.Add(item);
        }
        Status status = Status.RefreshAlgoProj ;
        LblStatus = $"Status:" + status.GetDescription();
    }

    /// <summary>
    /// Selected Project RefreshCommand
    /// </summary>
    public void RefreshSelectProj()
    {
        List<AlgoProjInfo> list = mOperation.GetInfo<AlgoProjInfo>(x => x.ProjName == ExistingSelectedProj.ProjName );
        if (list.Count > 0)
        {
            ExistingSelectedProj = list[0];
        }
        SelectedProjName = ExistingSelectedProj.ProjName;
        SelectedProjSavePath = ExistingSelectedProj.SavePath;
        SelectedProjIterNum = ExistingSelectedProj.IterNum.ToString();
        SelectedProjIterId = ExistingSelectedProj.IterId.ToString();
        SelectedProjDisplayId = ExistingSelectedProj.DisplayId.ToString();
        SelectedProjDownloadId = ExistingSelectedProj.DownloadId.ToString();
        SelectedProjUploadId = ExistingSelectedProj.UploadId.ToString();
        SelectedProjAlgoGenId = ExistingSelectedProj.AlgoGenId.ToString();
        SelectedProjAlgoRecevId = ExistingSelectedProj.AlgoRecevId.ToString();
        SelectedProjBarCodeId = ExistingSelectedProj.ConfigBarCodeId.ToString();
        SelectedProjExpId = ExistingSelectedProj.ExpId.ToString();
        SelectedProjBatchNum = ExistingSelectedProj.BatchNum.ToString();
        SelectedProjCreateTime = ExistingSelectedProj.CreateTime;
    }

    /// <summary>
    /// Add New Project Command
    /// </summary>
    [RelayCommand]
    public void AddProj()
    {
        // Input validation
        if (string.IsNullOrEmpty(NewProjName) || string.IsNullOrEmpty(NewProjIterNum) || string.IsNullOrEmpty(NewProjSavePath))
        {
            MessageBox.Show("Please complete the project information!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (!int.TryParse(NewProjIterNum, out int iterNum))
        {
            MessageBox.Show("Please enter a valid integer for iteration number!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (!int.TryParse(NewProjBatchNum, out int batchNum))
        {
            MessageBox.Show("Please enter a valid integer for batch number!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (!Directory.Exists(NewProjSavePath))
        {
            MessageBox.Show("Please select a valid save path!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        // Create new project
        var newProj = new AlgoProjInfo()
        {
            ProjName = NewProjName,
            CreateTime = DateTime.Now.ToString("yy-MM-dd HH:mm"),
            IterNum = iterNum,
            BatchNum = batchNum,
            IterId = 1,
            DisplayId = 0,
            UploadId = 0,
            DownloadId = 0,
            AlgoGenId = 0,
            AlgoRecevId = 0,
            ConfigBarCodeId = 0,
            ExpId = 0,
            SavePath = NewProjSavePath
        };
        // Check for duplicate project names
        List<AlgoProjInfo> list = mOperation.GetInfo<AlgoProjInfo>();
        foreach (var proj in list)
        {
            if (proj.ProjName == newProj.ProjName)
            {
                MessageBox.Show("A project with the same name already exists. Please choose a different name.", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
        }
        //DB insert
        mOperation.AddInfo(newProj);
        MessageBox.Show("Create new project success!", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
        Status status = Status.AddAlgoProj;
        LblStatus = $"Status:" + status.GetDescription();
    }

    /// <summary>
    /// Select File Path Command
    /// </summary>
    [RelayCommand]
    public void NewProjSaveDoubleClick()
    {
        System.Windows.Forms.FolderBrowserDialog folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog();
        if (folderBrowserDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
        {
            NewProjSavePath = folderBrowserDialog.SelectedPath;
        }
    }

    /// <summary>
    /// Select Existing Project Command
    /// </summary>
    [RelayCommand]
    public void SelectProj()
    {
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show(" Please select an existing file.", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
        if (ExistingSelectedProj != null)
        {
            // Load project data
            SelectedProjName = ExistingSelectedProj.ProjName;
            SelectedProjSavePath = ExistingSelectedProj.SavePath;
            SelectedProjIterNum = ExistingSelectedProj.IterNum.ToString();
            SelectedProjIterId = ExistingSelectedProj.IterId.ToString();
            SelectedProjDisplayId = ExistingSelectedProj.DisplayId.ToString();
            SelectedProjDownloadId = ExistingSelectedProj.DownloadId.ToString();
            SelectedProjUploadId = ExistingSelectedProj.UploadId.ToString();
            SelectedProjAlgoGenId = ExistingSelectedProj.AlgoGenId.ToString();
            SelectedProjAlgoRecevId = ExistingSelectedProj.AlgoRecevId.ToString();
            SelectedProjBarCodeId = ExistingSelectedProj.ConfigBarCodeId.ToString();
            SelectedProjExpId = ExistingSelectedProj.ExpId.ToString();
            SelectedProjBatchNum = ExistingSelectedProj.BatchNum.ToString();
            SelectedProjCreateTime = ExistingSelectedProj.CreateTime;
            // Load related experiment label
            LblIterIdAndIterNum = $"IterId/IterNum: {ExistingSelectedProj.IterId}/{ExistingSelectedProj.IterNum}";
            LblIterBatch = $"IterBatch: {ExistingSelectedProj.DownloadId}";
            if (ExistingSelectedProj.DownloadId > 0)
            {
                LblExpIdAndExpNum = $"ExpId/ExpRange: {ExistingSelectedProj.ExpId}/[{(ExistingSelectedProj.DownloadId - 1) * ExistingSelectedProj.BatchNum + 1}-{ExistingSelectedProj.DownloadId * ExistingSelectedProj.BatchNum}]";
            }
            //Update status
            Status status = Status.InitAlgoProj;
            LblStatus = $"Status:" + status.GetDescription();
            //Update experiment param
            BayesExperDataList.Clear();
            if (ExistingSelectedProj.DownloadId > 0)
            {
                List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId);
                foreach (var item in list)
                {
                    BayesExperDataList.Add(item);
                }
            }
            CurrentDisplaytIterId = ExistingSelectedProj.DownloadId;
            BthEnableRefresh();
            PlotLatestDisplayIdData();
        }
    }

    [RelayCommand]
    public void UnselectProj()
    {
        var result = MessageBox.Show(" Please confirm the unselect file.", "Info", MessageBoxButton.YesNo, MessageBoxImage.Question);
        if (result == MessageBoxResult.Yes)
        {
            if (ExistingSelectedProj != null)
            {
                ExistingSelectedProj = null;
                LblIterIdAndIterNum = $"IterId/IterNum: None";
                LblIterBatch = $"IterBatch: None";
                LblExpIdAndExpNum = $"ExpId/ExpRange: None";
                SelectedProjName = "";
                SelectedProjSavePath = "";
                SelectedProjIterNum = "";
                SelectedProjIterId = "";
                SelectedProjBarCodeId = "";
                SelectedProjUploadId = "";
                SelectedProjDownloadId = "";
                SelectedProjAlgoGenId = "";
                SelectedProjAlgoRecevId = "";
                SelectedProjBatchNum = "";
                SelectedProjExpId = "";
                SelectedProjCreateTime = "";

                BthEnableRefresh();
                BayesExperDataList.Clear();

                Status status = Status.UnSelectAlgoProj;
                LblStatus = $"Status:" + status.GetDescription();
                BthEnableRefresh();
                ClearPlot();
            }
            else
            {
                MessageBox.Show("No project is currently selected.", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            }
        }
    }

    [RelayCommand]
    public void DeleteProj()
    {
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show("Please select the project you want to remove", "Info", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        var result = MessageBox.Show(
            $"Please confirm the project you want to remove [{ExistingSelectedProj.ProjName}]",
            "Info",
            MessageBoxButton.YesNo,
            MessageBoxImage.Question);

        if (result == MessageBoxResult.Yes)
        {
            bool dbResult = mOperation.DeleteInfo(ExistingSelectedProj);
            if (dbResult)
            {
                ExistingProjList.Remove(ExistingSelectedProj);
                ExistingSelectedProj = null;
                SelectedProjName = "";
                SelectedProjSavePath = "";
                SelectedProjIterNum = "";
                SelectedProjIterId = "";
                SelectedProjUploadId = "";
                SelectedProjDownloadId = "";
                SelectedProjAlgoGenId = "";
                SelectedProjAlgoRecevId = "";
                SelectedProjBarCodeId = "";
                SelectedProjBatchNum = "";
                SelectedProjExpId = "";
                SelectedProjCreateTime = "";
                LblIterIdAndIterNum = "IterId/IterNum: None";
                LblIterBatch = "IterBatch: None";
                LblExpIdAndExpNum = "ExpId-ExpNum: None";
                LblStatus = "Status: None";


                Status status = Status.DeleteAlgoProj;
                LblStatus = $"Status:" + status.GetDescription();
                BthEnableRefresh();
                ClearPlot();
                MessageBox.Show("Delete the project sucessful！", "Inform", MessageBoxButton.OK, MessageBoxImage.Information);

            }
            else
            {
                MessageBox.Show("Delete the project failed！", "Warning", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
    }

    #endregion

    #region  Iter Command

    /// <summary>
    /// Next Iter Command
    /// </summary>
    [RelayCommand]
    private void IterBatchNext()
    {
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        bool dbResult = mOperation.UpdateInfo(ExistingSelectedProj);
        if (dbResult)
        {
            CurrentDisplaytIterId++;
            LblIterBatch = $"IterBatch: {CurrentDisplaytIterId}";
            if (CurrentDisplaytIterId == ExistingSelectedProj.DownloadId)
            {
                LblExpIdAndExpNum = $"ExpId/ExpRange: {ExistingSelectedProj.ExpId}/[{(CurrentDisplaytIterId - 1) * ExistingSelectedProj.BatchNum + 1}-{CurrentDisplaytIterId * ExistingSelectedProj.BatchNum}]";
            }
            else
            {
                LblExpIdAndExpNum = $"ExpId/ExpRange: None/[{(CurrentDisplaytIterId - 1) * ExistingSelectedProj.BatchNum + 1}-{CurrentDisplaytIterId * ExistingSelectedProj.BatchNum}]";

            }
            BayesExperDataList.Clear();
            if (CurrentDisplaytIterId > 0)
            {
                List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == CurrentDisplaytIterId);
                foreach (var item in list)
                {
                    BayesExperDataList.Add(item);
                }
            }
            Status status = Status.NextBatchIter;
            LblStatus = $"Status:" + status.GetDescription();
            PlotCurrentDisplayIdData();
            BthEnableRefresh();
        }
        else
        {
            MessageBox.Show("Enter the next iteration failed!", "Warning", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    /// <summary>
    /// Prev Iter Command
    /// </summary>
    [RelayCommand]
    private void IterBatchPrev()
    {
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        bool dbResult = mOperation.UpdateInfo(ExistingSelectedProj);
        if (dbResult)
        {
            CurrentDisplaytIterId--;
            LblIterBatch = $"IterBatch: {CurrentDisplaytIterId}";
            LblExpIdAndExpNum = $"ExpId/ExpRange: None/[{(CurrentDisplaytIterId - 1) * ExistingSelectedProj.BatchNum + 1}-{CurrentDisplaytIterId * ExistingSelectedProj.BatchNum}]";
            BayesExperDataList.Clear();
            if (CurrentDisplaytIterId > 0)
            {
                List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == CurrentDisplaytIterId);
                foreach (var item in list)
                {
                    BayesExperDataList.Add(item);
                }
            }
            Status status = Status.PrevBatchIter;
            LblStatus = $"Status:" + status.GetDescription();
            PlotCurrentDisplayIdData();
            BthEnableRefresh();
        }
        else
        {
            MessageBox.Show("Enter the prev iteration failed!", "Warning", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    /// <summary>
    /// Bayes Exper DataGrid Double Click Event
    /// </summary>
    [RelayCommand]
    private void BayesDoubleClick()
    {
        if (BayesSelectedItem != null)
        {
            var window = new Window();
            ExpData expData = new ExpData(BayesSelectedItem);
            expData.AdhensionEvent += my_AdhensionEvent;
            window.Content = expData;
            window.Width = 1000;
            window.Height = 700;
            window.Show();
            expData.Tag = window;
        }
    }

    /// <summary>
    /// Adhension Event Handler
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="arge"></param>
    void my_AdhensionEvent(object sender, AdhensionEventArges arge)
    {
        AdensionData test = arge.value;
        List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId && x.ExpID == test.expID);
        list[0].Adhesion = test.Adhension;
        list[0].DataCheck = true;
        bool dbResult = mOperation.UpdateInfo(list, x => new { x.Adhesion, x.DataCheck });
        if (dbResult)
        {
            MessageBox.Show("Add Adhension Value success!", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
        }

        BayesExperDataList.Clear();
        list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId);
        foreach (var item in list)
        {
            BayesExperDataList.Add(item);
        }
        if (sender is ExpData expData)
        {
            if (expData.Tag is Window parentWindow)
            {
                parentWindow.Close();
            }
        }


        Status status = Status.SubmitAdhensionValue;
        LblStatus = $"Status:" + status.GetDescription();
        BthEnableRefresh();
    }

    ///// <summary>
    ///// Download Parameter Command
    ///// </summary>
    //[RelayCommand]
    //private void DownloadParameter()
    //{
    //    //需要改成多线程/异步方式
    //    var result = MessageBox.Show("Confirm download parameters?", "Inform", MessageBoxButton.YesNo, MessageBoxImage.Information);
    //    if (result == MessageBoxResult.No)
    //    {
    //        return;
    //    }
    //    if (ExistingSelectedProj == null)
    //    {
    //        MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
    //        return;
    //    }
    //    if (ExistingSelectedProj.DownloadId >= ExistingSelectedProj.IterId)
    //    {
    //        MessageBox.Show("No more parameters to download!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
    //        return;
    //    }
    //    //Python Script to get new parameter
    //    string pythonPath = @"C:\Users\34236\anaconda3\envs\metal\python.exe";
    //    string scriptPath = @"c:\Users\34236\Desktop\MyAiProj\bayes.py";
    //    string cmd = "GetIterIdSample";
    //    string projName = ExistingSelectedProj.ProjName;
    //    int iterId = ExistingSelectedProj.IterId;
    //    string arguments = $"\"{scriptPath}\" {cmd} {projName} {iterId}";

    //    string output = string.Empty;
    //    string error = string.Empty;

    //    try
    //    {
    //        ProcessStartInfo startInfo = new ProcessStartInfo
    //        {
    //            FileName = pythonPath,
    //            Arguments = arguments,
    //            UseShellExecute = false,
    //            RedirectStandardOutput = true,
    //            RedirectStandardError = true,
    //            CreateNoWindow = true
    //        };

    //        using (Process process = Process.Start(startInfo))
    //        {
    //            var outputTask = process.StandardOutput.ReadToEndAsync();
    //            var errorTask = process.StandardError.ReadToEndAsync();

    //            process.WaitForExit();

    //            output = outputTask.Result;
    //            error = errorTask.Result;
    //        }
    //        if (!string.IsNullOrEmpty(error))
    //        {
    //        }

    //    }
    //    catch (Exception ex)
    //    {
    //    }
    //    finally
    //    {

    //    }

    //    MessageBox.Show("【Message】\n" + result);


    //    //需要重复读取AlogProjInfo的数据库信息，判断AlgoGenId是否==IterId，若满足要求，进行后续操作，需完善逻辑
    //    RefreshSelectProj();
    //    if(ExistingSelectedProj.AlgoGenId ==ExistingSelectedProj.IterId)
    //    {

    //    }


    //    //DB Operate 
    //    ExistingSelectedProj.DownloadId++;
    //    ExistingSelectedProj.ExpId = ExistingSelectedProj.DownloadId * ExistingSelectedProj.BatchNum;
    //    bool dbResult = mOperation.UpdateInfo(ExistingSelectedProj);
    //    if (dbResult)
    //    {
    //        SelectedProjDownloadId = ExistingSelectedProj.DownloadId.ToString();
    //        SelectedProjExpId = ExistingSelectedProj.ExpId.ToString();
    //        LblIterIdAndIterNum = $"IterId/IterNum: {ExistingSelectedProj.IterId}/{ExistingSelectedProj.IterNum}";
    //        LblExpIdAndExpNum = $"ExpId/ExpRange: {ExistingSelectedProj.ExpId}/[{(ExistingSelectedProj.DownloadId - 1) * ExistingSelectedProj.BatchNum + 1}-{ExistingSelectedProj.DownloadId * ExistingSelectedProj.BatchNum}]";
    //        BayesExperDataList.Clear();
    //        List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId);
    //        foreach (var item in list)
    //        {
    //            BayesExperDataList.Add(item);
    //        }
    //        CurrentDisplaytIterId = ExistingSelectedProj.DownloadId;
    //        Status status = Status.DownloadAlgorithm;
    //        LblStatus = $"Status:" + status.GetDescription();
    //        LblIterBatch = $"IterBatch: {ExistingSelectedProj.DownloadId}";
    //        BthEnableRefresh();
    //        MessageBox.Show("Download parameter success！", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
    //    }
    //    else
    //    {
    //        MessageBox.Show("Download parameter failed!", "Warning", MessageBoxButton.OK, MessageBoxImage.Error);
    //    }
    //}

    /// <summary>
    /// Download Parameter Command
    /// </summary>
    [RelayCommand]
    private async Task DownloadParameter()
    {
        try
        {
            var result = MessageBox.Show("Confirm download parameters?", "Inform", MessageBoxButton.YesNo, MessageBoxImage.Information);
            if (result == MessageBoxResult.No)
            {
                return;
            }

            // 检查项目是否已选择
            if (ExistingSelectedProj == null)
            {
                MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // 检查是否有可下载的参数
            if (ExistingSelectedProj.DownloadId >= ExistingSelectedProj.IterId)
            {
                MessageBox.Show("No more parameters to download!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            // 异步执行Python脚本
            string pythonPath = @"C:\Users\34236\anaconda3\envs\metal\python.exe";
            string scriptPath = @"c:\Users\34236\Desktop\MyAiProj\bayes.py";
            string cmd = "GetIterIdSample";
            string projName = ExistingSelectedProj.ProjName;
            int iterId = ExistingSelectedProj.IterId;
            string arguments = $"\"{scriptPath}\" {cmd} {projName} {iterId}";

            // 异步执行Python脚本
            var (output, error, success) = await ExecutePythonScriptAsync(pythonPath, arguments);

            if (!success)
            {
                MessageBox.Show($"Failed to execute Python script: {error}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // 轮询检查AlgoGenId是否已设置
            bool isAlgoGenIdSet = await PollAlgoGenIdSetAsync(ExistingSelectedProj, 10, 2000); // 最多检查10次，每次间隔2秒

            if (!isAlgoGenIdSet)
            {
                MessageBox.Show("Failed to generate parameters within timeout period.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // 刷新项目信息
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                RefreshSelectProj();
            });

            // 再次检查AlgoGenId是否正确设置
            if (ExistingSelectedProj.AlgoGenId != ExistingSelectedProj.IterId)
            {
                MessageBox.Show("Parameters generation failed. Please try again.", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // 执行数据库操作
            bool dbResult = await Task.Run(() =>
            {
                ExistingSelectedProj.DownloadId++;
                ExistingSelectedProj.ExpId = ExistingSelectedProj.DownloadId * ExistingSelectedProj.BatchNum;
                return mOperation.UpdateInfo(ExistingSelectedProj);
            });

            if (dbResult)
            {
                // 在UI线程更新UI元素
                await Application.Current.Dispatcher.InvokeAsync(() =>
                {
                    SelectedProjDownloadId = ExistingSelectedProj.DownloadId.ToString();
                    SelectedProjExpId = ExistingSelectedProj.ExpId.ToString();
                    LblIterIdAndIterNum = $"IterId/IterNum: {ExistingSelectedProj.IterId}/{ExistingSelectedProj.IterNum}";
                    LblExpIdAndExpNum = $"ExpId/ExpRange: {ExistingSelectedProj.ExpId}/[{(ExistingSelectedProj.DownloadId - 1) * ExistingSelectedProj.BatchNum + 1}-{ExistingSelectedProj.DownloadId * ExistingSelectedProj.BatchNum}]";

                    // 加载实验数据
                    BayesExperDataList.Clear();
                    List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId);
                    foreach (var item in list)
                    {
                        BayesExperDataList.Add(item);
                    }

                    CurrentDisplaytIterId = ExistingSelectedProj.DownloadId;
                    Status status = Status.DownloadAlgorithm;
                    LblStatus = $"Status:" + status.GetDescription();
                    LblIterBatch = $"IterBatch: {ExistingSelectedProj.DownloadId}";
                    BthEnableRefresh();

                    MessageBox.Show("Download parameter success！", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                });
            }
            else
            {
                MessageBox.Show("Download parameter failed!", "Warning", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }
        catch (Exception ex)
        {
            // 记录异常日志
            Console.WriteLine($"Exception during parameter download: {ex.Message}\n{ex.StackTrace}");
            MessageBox.Show($"An error occurred: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
        }
    }

    // 替换 ExecutePythonScriptAsync 方法中的 WaitForExitAsync 为传统的异步等待方式
    private async Task<(string Output, string Error, bool Success)> ExecutePythonScriptAsync(string pythonPath, string arguments)
    {
        try
        {
            using (var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = arguments,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                }
            })
            {
                process.Start();

                // 异步读取输出和错误
                var outputTask = process.StandardOutput.ReadToEndAsync();
                var errorTask = process.StandardError.ReadToEndAsync();

                // 等待进程完成（同步阻塞，但不会引发CS1061错误）
                await Task.Run(() => process.WaitForExit());

                // 获取结果
                string output = await outputTask;
                string error = await errorTask;

                return (output, error, process.ExitCode == 0);
            }
        }
        catch (Exception ex)
        {
            return (string.Empty, ex.Message, false);
        }
    }

    /// <summary>
    /// 轮询检查AlgoGenId是否已设置
    /// </summary>
    private async Task<bool> PollAlgoGenIdSetAsync(dynamic project, int maxAttempts, int intervalMs)
    {
        int attempts = 0;

        while (attempts < maxAttempts)
        {
            await Task.Delay(intervalMs);

            // 在UI线程刷新项目信息并检查AlgoGenId
            bool isSet = await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                RefreshSelectProj();
                return ExistingSelectedProj.AlgoGenId == ExistingSelectedProj.IterId;
            });

            if (isSet)
            {
                return true;
            }

            attempts++;
        }

        return false;
    }



    /// <summary>
    /// Upload Parameter Command
    /// </summary>
    [RelayCommand]
    private void UploadParameter()
    {
        var result = MessageBox.Show("Confirm upload parameters?", "Inform", MessageBoxButton.YesNo, MessageBoxImage.Information);
        if (result == MessageBoxResult.No)
        {
            return;
        }
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        if (ExistingSelectedProj.UploadId >= ExistingSelectedProj.IterId)
        {
            MessageBox.Show("No more parameters to upload!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        //Check Value
        foreach (var item in BayesExperDataList)
        {
            if (item.Barcode == null || !item.DataCheck)
            {
                MessageBox.Show("Please check all data are ready!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }
        }
        //Use python script to plot data
        PythonScriptPlot();
    }


    /// <summary>
    /// Config BarCode Command
    /// </summary>
    [RelayCommand]
    private void ConfigBarCode()
    {
        if (BayesExperDataList.Count > 0)
        {
            var window = new Window();
            ConfigBarCode config = new ConfigBarCode(BayesExperDataList);
            config.ItemEvent += my_ItemEvent;
            window.Content = config;
            window.Show();
            config.Tag = window;
        }
    }

    /// <summary>
    /// Item Event Handler
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="arge"></param>
    void my_ItemEvent(object sender, ItemEventArges arge)
    {
        BindingList<BayesExperData> BayesData = new BindingList<BayesExperData>();
        BayesData = arge.value;
        List<BayesExperData> list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId);
        foreach (var item in BayesData)
        {
            var matchedItem = list.Find(b => b.ExpID == item.ExpID);
            if (matchedItem != null)
            {
                matchedItem.Barcode = item.Barcode;
            }
        }
        mOperation.UpdateInfo(list, x => new { x.Barcode });

        BayesExperDataList.Clear();
        list = mOperation.GetInfo<BayesExperData>(x => x.ProjName == ExistingSelectedProj.ProjName && x.IterId == ExistingSelectedProj.DownloadId);
        foreach (var item in list)
        {
            BayesExperDataList.Add(item);
        }
        ExistingSelectedProj.ConfigBarCodeId++;
        SelectedProjBarCodeId = ExistingSelectedProj.ConfigBarCodeId.ToString();
        Status status = Status.ConfigBarCodeSuccess;
        LblStatus = $"Status:" + status.GetDescription();
        LblIterBatch = $"IterBatch: {ExistingSelectedProj.DownloadId}";
        bool dbResult = mOperation.UpdateInfo(ExistingSelectedProj);
        if (dbResult)
        {
            MessageBox.Show("Config BarCode success!", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
            BthEnableRefresh();
        }
        if (sender is ConfigBarCode configBarCode)
        {
            if (configBarCode.Tag is Window parentWindow)
            {
                parentWindow.Close();
            }
        }
    }


    #endregion

    #region PLOT Command


    private async void PythonScriptPlot()
    {
        try
        {
            string pythonPath = @"C:\Users\34236\anaconda3\envs\metal\python.exe";
            string scriptPath = @"c:\Users\34236\Desktop\MyAiProj\DataVisual.py";
            string projName = ExistingSelectedProj.ProjName;
            string outputDir = ExistingSelectedProj.SavePath;
            int iterNum = ExistingSelectedProj.IterId;
            string arguments = $"\"{scriptPath}\" --proj_name {projName} --iter_num {iterNum} --output_dir {outputDir}";

            string result = await Task.Run(() =>
            {
                string output = string.Empty;
                string error = string.Empty;

                try
                {
                    ProcessStartInfo startInfo = new ProcessStartInfo
                    {
                        FileName = pythonPath,
                        Arguments = arguments,
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    };

                    using (Process process = Process.Start(startInfo))
                    {
                        var outputTask = process.StandardOutput.ReadToEndAsync();
                        var errorTask = process.StandardError.ReadToEndAsync();

                        process.WaitForExit();

                        output = outputTask.Result;
                        error = errorTask.Result;
                    }
                    if (!string.IsNullOrEmpty(error))
                    {
                        return $"【ERROR】 {error}";
                    }

                    return output;
                }
                catch (Exception ex)
                {
                    return $"【EXCEPTION】 {ex.Message}";
                }
                finally
                {

                }
            });

            MessageBox.Show("【Message】\n" + result);
        }
        catch (Exception ex)
        {
            MessageBox.Show("【Error】\n" + ex.Message);
        }
        finally
        {
            //Get Latest Display
            ExistingSelectedProj.DisplayId++;
            SelectedProjDisplayId = ExistingSelectedProj.DisplayId.ToString();
            mOperation.UpdateInfo(ExistingSelectedProj);
            PlotLatestDisplayIdData();

            //Update INDEX
            ExistingSelectedProj.UploadId++;
            ExistingSelectedProj.IterId++;
            SelectedProjUploadId = ExistingSelectedProj.UploadId.ToString();
            SelectedProjIterId = ExistingSelectedProj.IterId.ToString();
            LblIterIdAndIterNum = $"IterId/IterNum: {ExistingSelectedProj.IterId}/{ExistingSelectedProj.IterNum}";
            mOperation.UpdateInfo(ExistingSelectedProj);
            Status status = Status.UploadAlgothrim;
            LblStatus = $"Status:" + status.GetDescription();
            LblIterBatch = $"IterBatch: {ExistingSelectedProj.DownloadId}";
            BthEnableRefresh();
            MessageBox.Show("Upload data success!", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }


    /// <summary>
    /// Plot Latest DisplayId Data
    /// </summary>
    private void PlotLatestDisplayIdData()
    {
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        DataPlotVisibility = Visibility.Visible;
        string BezierCurve, Umap, Pareto, IterValue;
        LblDataVisualization = "Data Visualization - " + ExistingSelectedProj.ProjName + " - IterID: " + ExistingSelectedProj.DisplayId.ToString();
        if (BezierComboxSelectedIndex == 1)
        {
            BezierCurve = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Uniformity.png";
        }
        else if (BezierComboxSelectedIndex == 2)
        {
            BezierCurve = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Adhesion.png";
        }
        else
        {
            BezierCurve = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Coverage.png";
        }
        BezierCurveImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, BezierCurve);

        if(UmapComboxSelectedIndex ==1)
        {
            Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_UniformityUmap.png";
        }
        else if (UmapComboxSelectedIndex == 2)
        {
            Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_AdhesionUmap.png";
        }
        else
        {
            Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_CoverageUmap.png";
        }
        UmapImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Umap);

        Pareto = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Pareto.png";
        ParetoImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Pareto);

        IterValue = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_IterValue.png";
        IterValueImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, IterValue);

    }

    /// <summary>
    /// Plot Current DisplayId Data
    /// </summary>
    private void PlotCurrentDisplayIdData()
    {
        if (ExistingSelectedProj == null)
        {
            MessageBox.Show("Please select an existing project first!", "Warning", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }
        DataPlotVisibility = Visibility.Visible;
        string BezierCurve,Umap, Pareto, IterValue;
        if (CurrentDisplaytIterId < ExistingSelectedProj.DisplayId)
        {
            LblDataVisualization = "Data Visualization - " + ExistingSelectedProj.ProjName + " - IterID: " + CurrentDisplaytIterId.ToString();
            if(BezierComboxSelectedIndex == 1)
            {
                BezierCurve = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Uniformity.png";
            }
            else if(BezierComboxSelectedIndex == 2)
            {
                BezierCurve = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Adhesion.png";
            }
            else
            {          
                BezierCurve = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Coverage.png";
            }

            BezierCurveImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, BezierCurve);

            if (UmapComboxSelectedIndex == 1)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_UniformityUmap.png";
            }
            else if (UmapComboxSelectedIndex == 2)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_AdhesionUmap.png";
            }
            else
            {
                Umap = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_CoverageUmap.png";
            }
            UmapImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Umap);

            Pareto = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Pareto.png";
            ParetoImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Pareto);

            IterValue = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_IterValue.png";
            IterValueImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, IterValue);
        }
        else
        {
            LblDataVisualization = "Data Visualization - " + ExistingSelectedProj.ProjName + " - IterID: " + ExistingSelectedProj.DisplayId.ToString();
            if (BezierComboxSelectedIndex == 1)
            {
                BezierCurve = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Uniformity.png";
            }
            else if (BezierComboxSelectedIndex == 2)
            {
                BezierCurve = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Adhesion.png";
            }
            else
            {
                
                BezierCurve = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Coverage.png";
            }
            BezierCurveImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, BezierCurve);

            if (UmapComboxSelectedIndex == 1)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_UniformityUmap.png";
            }
            else if (UmapComboxSelectedIndex == 2)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_AdhesionUmap.png";
            }
            else
            {
                Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_CoverageUmap.png";
            }
            UmapImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Umap);

            Pareto = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_Pareto.png";
            ParetoImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Pareto);

            IterValue = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_IterValue.png";
            IterValueImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, IterValue);
        }
    }

    /// <summary>
    /// Clear Plot Data
    /// </summary>
    private void ClearPlot()
    {
        DataPlotVisibility = Visibility.Hidden;
        LblDataVisualization = "Data Visualization";
        BezierCurveImagePath = "";
        UmapImagePath = "";
        ParetoImagePath = "";
        IterValueImagePath = "";
    }

    /// <summary>
    /// Bezier Combox Selection Changed Event
    /// </summary>
    [RelayCommand]
    private void BezierComboxSelectionChanged()
    {
        string BezierCurve;
        if (BezierComboxSelectedIndex == 1)
        {
            BezierCurve = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Uniformity.png";
        }
        else if (BezierComboxSelectedIndex == 2)
        {
            BezierCurve = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Adhesion.png";
        }
        else
        {
            BezierCurve = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_Coverage.png";
        }
        BezierCurveImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, BezierCurve);
    }

    [RelayCommand]
    private void UmapComboxSelectionChanged()
    {
        string Umap;
        if (CurrentDisplaytIterId < ExistingSelectedProj.DisplayId)
        {
     

            if (UmapComboxSelectedIndex == 1)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_UniformityUmap.png";
            }
            else if (UmapComboxSelectedIndex == 2)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_AdhesionUmap.png";
            }
            else
            {
                Umap = ExistingSelectedProj.ProjName + "_" + CurrentDisplaytIterId.ToString() + "_CoverageUmap.png";
            }
            UmapImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Umap);         
        }
        else
        {
            if (UmapComboxSelectedIndex == 1)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_UniformityUmap.png";
            }
            else if (UmapComboxSelectedIndex == 2)
            {
                Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_AdhesionUmap.png";
            }
            else
            {
                Umap = ExistingSelectedProj.ProjName + "_" + ExistingSelectedProj.DisplayId.ToString() + "_CoverageUmap.png";
            }
            UmapImagePath = System.IO.Path.Combine(ExistingSelectedProj.SavePath, Umap);
        }
    }

    #endregion

}