using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace UI
{
    /// <summary>
    /// A simple game-over overlay dialog window.
    /// </summary>
    public partial class GameOverWindow : Window
    {
        /// <summary>
        /// Creates a new GameOverWindow with the given winner's name.
        /// </summary>
        /// <param name="winnerName"></param>
        public GameOverWindow(string winnerName)
        {
            InitializeComponent();
            _winnerAcclaim.Text = winnerName + _winnerAcclaim.Text;
        }

        /// <summary>
        /// Close the game's window and creates a new main window to create a new game.
        /// Note: duplicates HomePage.xaml.cs#NewGameClicked, could be moved to a Command.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NewGameClicked(object sender, RoutedEventArgs e)
        {
            var window = new MainWindow();
            var gameWindow = Application.Current.MainWindow;
            Application.Current.MainWindow = window;
            window.Show();
            gameWindow.Close();
            Close();
        }

        /// <summary>
        /// Exits the game.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ExitButtonClick(object sender, RoutedEventArgs e)
        {
            Application.Current.MainWindow.Close();
            Close();
        }
    }
}
