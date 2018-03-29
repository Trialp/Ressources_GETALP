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
    /// Provides controls to choose the number of players.
    /// </summary>
    public partial class ChoosePlayerCountPage : Page
    {
        public ChoosePlayerCountPage()
        {
            InitializeComponent();
        }


        /// <summary>
        /// Go to the previous page.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BackButtonClick(object sender, RoutedEventArgs e)
        {
            NavigationService.GoBack();
        }

        /// <summary>
        /// Go to the next page.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NextButtonClick(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new ChoosePlayersPage(int.Parse(_numberOfPlayers.Text)));
        }
    }
}
