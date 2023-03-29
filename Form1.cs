using OpenCvSharp;

namespace camera_measure
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.Multiselect = true;
            ofd.ShowDialog();
            var files= ofd.FileNames;
            //Cv2.ImRead(files[0]);
            caliberation.Run(files);
        }
    }
}