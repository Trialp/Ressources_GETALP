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
    /// Interaction logic for InGameMenuSavePage.xaml
    /// </summary>
    public partial class InGameMenuSavePage : Page
    {
        private Game _game;
        public InGameMenuSavePage(Game currentGame)
        {
            _game = currentGame;
            InitializeComponent();
        }

        /// <summary>
        /// Navigates to the previous page.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BackClicked(object sender, RoutedEventArgs e)
        {
            NavigationService.GoBack();
        }

        /// <summary>
        /// Saves the game to a file and show a success message.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void SaveClicked(object sender, RoutedEventArgs e)
        {
            try
            {
                GameBuilder.SaveGame(_gameName.Text, _game);
                Window.GetWindow(this).Close();
            }
            catch(Exception ex)
            {
                NavigationService.Navigate(new InGameMenuErrorPage());
            }
        }
    }
}
