/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * NIST-developed software is provided by NIST as a public service. You may
 * use, copy and distribute copies of the software in any medium, provided that
 * you keep intact this entire notice. You may improve, modify and create
 * derivative works of the software or any portion of the software, and you may
 * copy and distribute such modifications or works. Modified works should carry
 * a notice stating that you changed the software and should note the date and
 * nature of any such change. Please explicitly acknowledge the National
 * Institute of Standards and Technology as the source of the software.
 * 
 * NIST-developed software is expressly provided "AS IS." NIST MAKES NO
 * WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF
 * LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST
 * NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
 * UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST
 * DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE
 * SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE
 * CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.
 * 
 * You are solely responsible for determining the appropriateness of using and
 * distributing the software and you assume all risks associated with its use,
 * including but not limited to the risks and costs of program errors,
 * compliance with applicable laws, damage to or loss of data, programs or
 * equipment, and the unavailability or interruption of operation. This
 * software is not intended to be used in any situation where a failure could
 * cause risk of injury or damage to property. The software developed by NIST
 * employees is not subject to copyright protection within the United States.
 * 
 * Author: Evan Black <evan.black@nist.gov>
 */


#include <ns3/core-module.h>
#include <ns3/energy-module.h>
#include <ns3/mobility-module.h>
#include <ns3/psc-module.h>

using namespace ns3;
using namespace psc;

NS_LOG_COMPONENT_DEFINE ("UavEnergyMobility");

void
RemainingEnergy (double oldValue, double remainingEnergy)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds ()
                   << "s Current remaining energy = " << remainingEnergy << "J");
}

void
EnergyDepleted (Ptr<ConstantVelocityMobilityModel> mobilityModel, Ptr<const UavMobilityEnergyModel> energyModel)
{
  NS_LOG_UNCOND (Simulator::Now ().GetSeconds () << "s ENERGY DEPLETED");
  auto currentPosition = mobilityModel->GetPosition ();
  // Stop & Drop the UAV
  mobilityModel->SetPosition ({ currentPosition.x, currentPosition.y, 0});
  Simulator::Stop ();
}


int
main (int argc, char *argv[])
{
  const Vector MOVE_X_VELOCITY (1, 0, 0);
  const Vector STOPPED_VELOCITY (0, 0, 0);
  const Vector ASCEND_VELOCITY (0, 0, 1);
  const Vector DESCEND_VELOCITY (0, 0, -1);
  const double INITIAL_ENERGY = 356400; //https://www.nsnam.org/wiki/Li-Ion_model_fitting

  UavMobilityEnergyModelHelper helper;

  helper.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
  // helper.SetEnergySource("ns3::BasicEnergySource", 
  //                        "BasicEnergySourceInitialEnergyJ",
  //                        DoubleValue (INITIAL_ENERGY));

    // helper.SetEnergySource("ns3::LiIonEnergySource",
    //                      "LiIonEnergySourceInitialEnergyJ",
    //                      DoubleValue (INITIAL_ENERGY));

     helper.SetEnergySource("ns3::LiIonEnergySource",
                         "LiIonEnergySourceInitialEnergyJ",
                         DoubleValue (INITIAL_ENERGY),
                         "InitialCellVoltage",
                         DoubleValue (3.45),
                         "NominalCellVoltage",
                         DoubleValue (3.3),
                         "ExpCellVoltage",
                         DoubleValue (3.55),
                          "RatedCapacity",
                         DoubleValue (30),
                         "NomCapacity",
                         DoubleValue (27),
                         "ExpCapacity",
                         DoubleValue (15));

      // helper.SetEnergySource("ns3::LiIonEnergySource",
      //                    "InternalResistance",
      //                    DoubleValue (0.145),
      //                    "TypCurrent",
      //                    DoubleValue (1));

  Ptr<Node> node = CreateObject<Node> ();
  DeviceEnergyModelContainer model = helper.Install (node);

  Ptr<ConstantVelocityMobilityModel> mobilityModel =
    node->GetObject<ConstantVelocityMobilityModel> ();

  //Ptr<BasicEnergySource> source = node->GetObject<BasicEnergySource> ();
  Ptr<LiIonEnergySource> source = node->GetObject<LiIonEnergySource> ();
  //source->SetInitialSupplyVoltage(22.8);

  source->TraceConnectWithoutContext ("RemainingEnergy",
                                      MakeCallback (RemainingEnergy));
  // NS_LOG_UNCOND("Decrease Remaining Energy 1000 Joules");
  //source->DecreaseRemainingEnergy(8000);

  model.Get(0)->TraceConnectWithoutContext ("EnergyDepleted",
                                           MakeBoundCallback (&EnergyDepleted, 
                                                              mobilityModel));
 Simulator::Schedule (Seconds (2),
                        &ConstantVelocityMobilityModel::SetVelocity,
                        mobilityModel, MOVE_X_VELOCITY);

  Simulator::Schedule (Seconds (5),
                        &ConstantVelocityMobilityModel::SetVelocity,
                        mobilityModel, STOPPED_VELOCITY);

  Simulator::Schedule (Seconds (6),
                        &ConstantVelocityMobilityModel::SetVelocity,
                        mobilityModel, ASCEND_VELOCITY);

  Simulator::Schedule (Seconds (8),
                        &ConstantVelocityMobilityModel::SetVelocity,
                        mobilityModel, DESCEND_VELOCITY);

  Simulator::Schedule (Seconds (10),
                       &ConstantVelocityMobilityModel::SetVelocity,
                       mobilityModel, STOPPED_VELOCITY);

  Simulator::Stop (Seconds (11));
  Simulator::Run ();
  Simulator::Destroy ();
}
