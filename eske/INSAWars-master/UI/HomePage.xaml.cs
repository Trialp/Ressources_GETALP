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

namespace UI
{
    /// <summary>
    /// Provides buttons to start a new game, load an existing game or exit the application.
    /// </summary>
    public partial class HomePage : Page
    {
        public HomePage()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Navigates to a new page to create a new game.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void CreateGameClicked(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new ChoosePlayerCountPage());
        }

        /// <summary>
        /// Exits the game.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void ExitClicked(object sender, RoutedEventArgs e)
        {
            Application.Current.MainWindow.Close();
        }

        /// <summary>
        /// Navigates to a new page to load an existing game.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LoadGameClicked(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new LoadGamePage());
        }
    }
}
