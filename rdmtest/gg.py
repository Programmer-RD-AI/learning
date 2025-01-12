from graphviz import Digraph

# Create a new UML diagram
dot = Digraph(comment='Appointment Management System UML')

# Add model classes
dot.node('Appointment', '''Appointment
- appointmentId: int
- patient: Patient
- dermatologist: DermatologistModel
- treatment: Treatment
- time: Time
- invoice: Invoice
- status: String
+ setStatus(status: String): void
+ getInvoice(): Invoice
+ getAppointmentId(): int
''')

dot.node('Patient', '''Patient
- patientId: int
- name: String
- age: int
- address: String
- contactInfo: String
+ getPatientDetails(): Patient
''')

dot.node('DermatologistModel', '''DermatologistModel
- name: String
+ checkAvailability(time: Time): boolean
''')

dot.node('Treatment', '''Treatment
- treatmentId: int
- treatmentName: String
- cost: double
+ getTreatmentDetails(): Treatment
''')

dot.node('Invoice', '''Invoice
- invoiceId: int
- totalAmount: double
- amountPaid: double
+ makePayment(amount: double): void
+ isPaid(): boolean
''')

dot.node('Time', '''Time
- date: LocalDate
- startTime: LocalTime
+ getEndTime(): LocalTime
''')

dot.node('DermatologistRepository', '''DermatologistRepository
- dermatologists: List<DermatologistModel>
+ addTimeSlot(dermatologist: DermatologistModel, time: Time): boolean
''')

dot.node('AppointmentRepository', '''AppointmentRepository
- appointments: List<Appointment>
+ createAppointment(appointment: Appointment): void
+ listAppointments(): List<Appointment>
+ getAppointmentById(appointmentId: int): Appointment
+ cancelAppointmentById(appointmentId: int): boolean
+ updateAppointment(appointmentId: int, patient: Patient, dermatologist: DermatologistModel, treatment: Treatment, time: Time): boolean
+ getAppointmentsByPatientName(patientName: String): List<Appointment>
''')

dot.node('AppointmentService', '''AppointmentService
+ scheduleAppointment(): void
+ viewAllAppointments(): void
+ updateAppointment(): void
+ cancelAppointment(): void
+ completeAppointment(): void
+ searchAppointment(): void
''')

dot.node('DermatologistService', '''DermatologistService
+ selectDermatologist(): DermatologistModel
+ addTimeSlotForDermatologist(): void
''')

dot.node('TimeService', '''TimeService
+ getTimeDetails(dermatologist: DermatologistModel): Time
''')

dot.node('TreatmentService', '''TreatmentService
+ getTreatmentDetails(): Treatment
''')

dot.node('PatientService', '''PatientService
+ getPatientDetails(): Patient
''')

# Add relationships (associations)
dot.edge('Appointment', 'Patient', 'has')
dot.edge('Appointment', 'DermatologistModel', 'has')
dot.edge('Appointment', 'Treatment', 'has')
dot.edge('Appointment', 'Time', 'has')
dot.edge('Appointment', 'Invoice', 'has')
dot.edge('AppointmentRepository', 'Appointment', 'manages')
dot.edge('DermatologistRepository', 'DermatologistModel', 'manages')
dot.edge('DermatologistRepository', 'Time', 'manages')
dot.edge('AppointmentService', 'AppointmentRepository', 'uses')
dot.edge('AppointmentService', 'PatientService', 'uses')
dot.edge('AppointmentService', 'DermatologistService', 'uses')
dot.edge('AppointmentService', 'TreatmentService', 'uses')
dot.edge('AppointmentService', 'TimeService', 'uses')
dot.edge('DermatologistService', 'DermatologistRepository', 'uses')
dot.edge('DermatologistService', 'TimeService', 'uses')
dot.edge('TreatmentService', 'Treatment', 'uses')
dot.edge('PatientService', 'Patient', 'uses')
dot.edge('TimeService', 'Time', 'uses')

dot.attr(dpi='300')  # Set the DPI to 300 for better resolution
dot.attr(rankdir='LR', size='12,8')  # Increase the diagram size for more detail

dot.render(filename='./UML_Diagram_Complete2.png', format='png', cleanup=False)


