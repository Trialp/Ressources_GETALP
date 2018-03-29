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
using System.Windows.Navigation;
using System.Windows.Shapes;
using INSAWars.Game;

namespace UI
{
    /// <summary>
    /// Interaction logic for InGameMenuHomePage.xaml
    /// </summary>
    public partial class InGameMenuHomePage : Page
    {
        private Game _game;

        public InGameMenuHomePage()
        {
            InitializeComponent();
        }

        public Game CurrentGame
        {
            get;
            set;
        }

        /// <summary>
        /// Shows the user a new page to save the current game to a file.
        /// Not implemented yet.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void SaveGameClicked(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new InGameMenuSavePage(CurrentGame));
        }

        /// <summary>
        /// Exits the application.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ExitClicked(object sender, RoutedEventArgs e)
        {
            Application.Current.MainWindow.Close();
            Window.GetWindow(this).Close();
        }

        /// <summary>
        /// Creates a new Main Window to start a new game.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BackToMainMenuClicked(object sender, RoutedEventArgs e)
        {
            var window = new MainWindow();
            var gameWindow = Application.Current.MainWindow;
            Application.Current.MainWindow = window;
            window.Show();
            gameWindow.Close();
            Window.GetWindow(this).Close();
        }
    }
}
