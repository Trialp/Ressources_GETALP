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
using System.IO;

namespace UI
{
    /// <summary>
    /// Provides controls to load a game.
    /// </summary>
    public partial class LoadGamePage : Page
    {
        public LoadGamePage()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Go to previous page.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void BackClicked(object sender, RoutedEventArgs e)
        {
            NavigationService.GoBack();
        }

        /// <summary>
        /// Loads the game from a file and show the game window.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void LoadGameClicked(object sender, RoutedEventArgs e)
        {
            try
            {
                var game = GameBuilder.LoadGame(_gameName.Text);

                var main = Application.Current.MainWindow;
                var gameWindow = new GameWindow(game);
                Application.Current.MainWindow = gameWindow;
                main.Close();
                gameWindow.Show();
            }
            catch (Exception _)
            {
                Opacity = 0.5;
                var dialog = new LoadErrorWindow();
                dialog.ShowDialog();
                Opacity = 1;
            }
            
        }
    }
}
