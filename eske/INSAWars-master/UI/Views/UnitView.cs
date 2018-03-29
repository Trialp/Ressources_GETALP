using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.MVVM;
using System.ComponentModel;
using INSAWars.Units;

namespace UI.Views
{
    /// <summary>
    /// Defines a ViewModel for Unit.
    /// </summary>
    public class UnitView : ObservableObject
    {

        private int _attack;
        private int _defense;
        private int _hitPoints;
        private int _movementPoints; 
        private int _remainingHitPoints;
        private int _remainingMovementPoints;
        private string _type;

        public UnitView(Unit unit)
        {
            Unit = unit;
            Attack = unit.AttackTotal;
            Defense = unit.DefenseTotal;
            HitPoints = unit.HitPoints;
            MovementPoints = unit.MovementPoints;
            RemainingHitPoints = unit.RemainingHitPoints;
            RemainingMovementPoints = unit.RemainingMovementPoints;
            Type = unit.ToString();

            unit.PropertyChanged += new PropertyChangedEventHandler(delegate(object sender, PropertyChangedEventArgs args)
            {
                switch (args.PropertyName)
                {
                    case "RemainingMovementPoints":
                        RemainingMovementPoints = ((Unit)sender).RemainingMovementPoints;
                        break;
                    case "RemainingHitPoints":
                        RemainingHitPoints = ((Unit)sender).RemainingHitPoints;
                        break;
                    default:
                        break;
                }
            });
        }

        public Unit Unit
        {
            set;
            get;
        }

        public int Attack
        {
            get { return _attack; }
            set { SetProperty(ref _attack, value); }
        }

        public int Defense
        {
            get { return _defense; }
            set { SetProperty(ref _defense, value); }
        }

        public int HitPoints
        {
            get { return _hitPoints; }
            set { SetProperty(ref _hitPoints, value); }
        }

        public int MovementPoints
        {
            get { return _movementPoints; }
            set { SetProperty(ref _movementPoints, value); }
        }

        public int RemainingHitPoints
        {
            get { return _remainingHitPoints; }
            set { SetProperty(ref _remainingHitPoints, value); }
        }

        public int RemainingMovementPoints
        {
            get { return _remainingMovementPoints; }
            set { SetProperty(ref _remainingMovementPoints, value); }
        }

        public string Type
        {
            get { return _type; }
            set { SetProperty(ref _type, value); }
        }
    }
}
