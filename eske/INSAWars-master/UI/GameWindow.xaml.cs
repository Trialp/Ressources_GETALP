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
using INSAWars.Game;
using UI.Views;
using UI.Commands;
using System.Windows.Threading;

namespace UI
{
    /// <summary>
    /// Provides the main interaction logic between the GameControl (that is responsible for graphics rendering)
    /// and the game's general UI.
    /// </summary>
    ///
    public partial class GameWindow : Window
    {
        private Game _game;
        private GameView _gameView;
        private CommandState _state;
        private Dictionary<Player, Case> _lastSelectedCases;

        /// <summary>
        /// Provides a nice way of dealing with the UI's current state.
        /// </summary>
        private enum CommandState
        {
            Attacking, // The player can attack a case.
            Moving, // The player can move a unit.
            Selecting // The player can select a new case.
        };

        /// <summary>
        /// Creates a new GameWindow from the given Game.
        /// </summary>
        /// <param name="builder"></param>
        public GameWindow(Game game)
        {
            _game = game;
            _game.GameIsOver += GameIsOver;

            _gameView = new GameView(_game);
            
            InitializeComponent();
            InitializeGameControl();
            InitializeDataContexts();
            InitializeClock();
            InitializeLastSelectedCases();
            ResetUIState();
        }

        private void InitializeLastSelectedCases()
        {
            _lastSelectedCases = new Dictionary<Player, Case>();
            foreach (Player p in _game.Players)
            {
                _lastSelectedCases.Add(p, p.Units.First().Location);
            }

            _gameControl.SelectCase(_lastSelectedCases[_game.CurrentPlayer]);
            _gameControl.MoveVisibleMapToSelectedCase();
        }

        /// <summary>
        /// Initializes the game's clock.
        /// </summary>
        private void InitializeClock()
        {
            var initial = DateTime.Now;
            DispatcherTimer timer = new DispatcherTimer(new TimeSpan(0, 0, 1), DispatcherPriority.Normal, delegate
            {
                _clock.Content = DateTime.Now.Subtract(initial).ToString(@"hh\:mm\:ss");
            }, Dispatcher);
        }

        /// <summary>
        /// Initializes the data contexts of UI elements.
        /// </summary>
        private void InitializeDataContexts()
        {
            _turnLabel.DataContext = _gameView;
            _playerInformation.DataContext = new PlayerView(_game.CurrentPlayer);
        }

        /// <summary>
        /// Initializes the game control and subscribe to its events.
        /// </summary>
        private void InitializeGameControl()
        {
            _gameControl.Game = _game;
            _gameControl.CaseClicked += CaseClicked;
            _gameControl.CaseSelected += CaseSelected;
        }

        /// <summary>
        /// Displays a dialog with the winner's name.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void GameIsOver(object sender, GameOverEventArgs e)
        {
            var menu = new GameOverWindow(e.Winner.Name);
            Opacity = 0.5;
            menu.ShowDialog();
        }

        /// <summary>
        /// Resets the UI state to the basic state, which is selecting a case.
        /// </summary>
        private void ResetUIState()
        {
            _state = CommandState.Selecting;
            Cursor = Cursors.Arrow;
        }

       /// <summary>
       /// Provides a very basic finite-state machine to control the interactions
       /// with the game. When a case is clicked, there are several possible
       /// actions according to the UI's current state.
       /// </summary>
       /// <param name="sender"></param>
       /// <param name="e"></param>
        private void CaseClicked(object sender, CaseClickedEventArgs e)
        {
            switch (_state)
            {
                case CommandState.Selecting:
                    var select = new SelectCaseCommand(_game, _gameControl);
                    if (select.CanExectute(e.ClickedCase))
                    {
                        select.Execute(e.ClickedCase);
                    }
                    else
                    {
                        _gameControl.DisplayInvalidCommandOn(e.ClickedCase);
                    }

                    break;
                case CommandState.Moving:
                    var unitToMove = ((UnitView)_units.SelectedItem).Unit;
                    var move = new MoveUnitCommand(_game, unitToMove);

                    if (move.CanExectute(e.ClickedCase))
                    {
                        move.Execute(e.ClickedCase);
                        ResetUIState();
                        _gameControl.SelectCase(e.ClickedCase);
                    }
                    else
                    {
                        _gameControl.DisplayInvalidCommandOn(e.ClickedCase);
                    }

                    break;
                case CommandState.Attacking:
                    var attackWithUnit = ((UnitView)_units.SelectedItem).Unit;
                    var attack = new AttackCommand(attackWithUnit);

                    if (attack.CanExectute(e.ClickedCase))
                    {
                        attack.Execute(e.ClickedCase);
                        ResetUIState();
                        _gameControl.SelectCase(e.ClickedCase);
                    }
                    else
                    {
                        _gameControl.DisplayInvalidCommandOn(e.ClickedCase);
                    }
                    break;
                default:
                    _state = CommandState.Selecting;
                    CaseClicked(sender, e);
                    break;
            }
        }

        /// <summary>
        /// Updates data contexts when a new case is selected.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void CaseSelected(object sender, CaseSelectedEventArgs e)
        {
            _caseInformation.DataContext = _unitActions.DataContext = (e.IsDeselection ? null : new CaseView(_game, e.SelectedCase));
            _lastSelectedCases[_game.CurrentPlayer] = e.SelectedCase;
        }

        /// <summary>
        /// Provides keyboard interaction for the UI.
        /// </summary>
        /// <param name="e"></param>
        protected override void OnPreviewKeyDown(KeyEventArgs e)
        {
            e.Handled = true;
            switch (e.Key)
            {
                case Key.Escape:
                    var menu = new InGameMenuWindow(_game);
                    Opacity = 0.5;
                    menu.ShowDialog();
                    Opacity = 1;
                    break;
                case Key.Left:
                    if (_gameControl.CanMoveVisibleMapLeft())
                    {
                        _gameControl.MoveVisibleMapLeft();
                    }
                    break;
                case Key.Right:

                    if (_gameControl.CanMoveVisibleMapRight())
                    {
                        _gameControl.MoveVisibleMapRight();
                    }
                    break;
                case Key.Up:
                    if (_gameControl.CanMoveVisibleMapUp())
                    {
                        _gameControl.MoveVisibleMapUp();
                    }
                    break;
                case Key.Down:
                    if (_gameControl.CanMoveVisibleMapDown())
                    {
                        _gameControl.MoveVisibleMapDown();
                    }
                    break;
                case Key.Space:
                    _gameControl.MoveVisibleMapToSelectedCase();
                    break;
                default:
                    e.Handled = false;
                    break;
            }
        }

        /// <summary>
        /// Resets the UI's state when the user clicks with the mouse's right button.
        /// </summary>
        /// <param name="e"></param>
        protected override void OnPreviewMouseUp(MouseButtonEventArgs e)
        {
            if (e.ChangedButton == MouseButton.Right)
            {
                ResetUIState();
            }
        }

        private void NextTurnClicked(object sender, RoutedEventArgs e)
        {
            _game.NextTurn();
            _playerInformation.DataContext = new PlayerView(_game.CurrentPlayer);
            ResetUIState();

            if (_lastSelectedCases[_game.CurrentPlayer] != null)
            {
                _gameControl.SelectCase(_lastSelectedCases[_game.CurrentPlayer]);
                _gameControl.MoveVisibleMapToSelectedCase();
            }
            else
            {
                _gameControl.ClearCaseSelection();
            }
        }

        private void MoveClicked(object sender, RoutedEventArgs e)
        {
            _state = CommandState.Moving;
            Cursor = ((FrameworkElement)this.Resources["MoveCursor"]).Cursor;
        }

        private void AttackClicked(object sender, RoutedEventArgs e)
        {
            _state = CommandState.Attacking;
            Cursor = ((FrameworkElement)this.Resources["AttackCursor"]).Cursor;
        }

        private void BuildCityClicked(object sender, RoutedEventArgs e)
        {
            _state = CommandState.Selecting;
            Cursor = Cursors.Arrow;
            var builder = ((UnitView)_units.SelectedItem).Unit;
            _game.BuildCity(builder);
            _gameControl.ClearCaseSelection();
        }

        private void MakeTeacherClicked(object sender, RoutedEventArgs e)
        {
            var c = ((CaseView)_caseInformation.DataContext).Case;
            var command = new MakeTeacherCommand(c);

            if (command.CanExecute())
            {
                command.Execute();
            }
            else
            {
                _gameControl.DisplayInvalidCommandOn(_gameControl.SelectedCase);
            }
        }

        private void MakeStudentClicked(object sender, RoutedEventArgs e)
        {
            var c = ((CaseView)_caseInformation.DataContext).Case;
            var command = new MakeStudentCommand(c);

            if (command.CanExecute())
            {
                command.Execute();
            }
            else
            {
                _gameControl.DisplayInvalidCommandOn(_gameControl.SelectedCase);
            }
        }

        private void MakeHeadClicked(object sender, RoutedEventArgs e)
        {
            var c = ((CaseView)_caseInformation.DataContext).Case;
            var command = new MakeHeadCommand(c);

            if (command.CanExecute())
            {
                command.Execute();
            }
            else
            {
                _gameControl.DisplayInvalidCommandOn(_gameControl.SelectedCase);
            }
        }
    }
}
