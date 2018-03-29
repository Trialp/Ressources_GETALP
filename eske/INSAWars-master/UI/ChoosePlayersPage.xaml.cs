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
using INSAWars.Units;

namespace UI
{
    /// <summary>
    /// Provides controls to personnalize players' name and civilization.
    /// </summary>
    public partial class ChoosePlayersPage : Page
    {
        private int _playerCount;

        public ChoosePlayersPage(int playerCount)
        {
            _playerCount = playerCount;
            InitializeComponent();
            HideUnusedPlayers();
        }

        /// <summary>
        /// Hides the controls that we do not need, according to number
        /// of players the user previously selected.
        /// </summary>
        private void HideUnusedPlayers()
        {
            if (_playerCount < 3)
            {
                _playerThree.Visibility = Visibility.Hidden;
            }
            if (_playerCount < 4)
            {
                _playerFour.Visibility = Visibility.Hidden;
            }
        }

        /// <summary>
        /// Adds a player to the game builder if the control is visible, that is, if
        /// there is enough players to make use of this control.
        /// </summary>
        /// <param name="builder"></param>
        /// <param name="player"></param>
        /// <param name="name"></param>
        /// <param name="civilization"></param>
        private void AddPlayerToGameBuilder(GameBuilder builder, StackPanel player, TextBox name, ComboBox civilization)
        {
            if (player.Visibility == Visibility.Visible)
            {
                builder.AddPlayer(name.Text, civilization.Text);
            }
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
        /// Adds players to the game builder and navigate to the next page.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void NextButtonClick(object sender, RoutedEventArgs e)
        {
            var builder = new GameBuilder();
            AddPlayerToGameBuilder(builder, _playerOne, _playerOneName, _playerOneCivilization);
            AddPlayerToGameBuilder(builder, _playerTwo, _playerTwoName, _playerTwoCivilization);
            AddPlayerToGameBuilder(builder, _playerThree, _playerThreeName, _playerThreeCivilization);
            AddPlayerToGameBuilder(builder, _playerFour, _playerFourName, _playerFourCivilization);
            
            NavigationService.Navigate(new ChooseMapSizePage(builder));
        }
    }
}
