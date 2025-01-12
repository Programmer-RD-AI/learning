from graphviz import Digraph

# Create a new UML diagram with improved settings
uml_diagram = Digraph('Complete_UML_Class_Diagram', filename='./Final_UML_Diagram.png', format='png')
uml_diagram.attr(rankdir='TB')  # Top to bottom layout for better readability
uml_diagram.attr(splines='ortho')  # Orthogonal lines for cleaner appearance
uml_diagram.attr(nodesep='0.8')
uml_diagram.attr(ranksep='1.0')
uml_diagram.attr(dpi='300')

# UI Layer (View)
uml_diagram.node('AppointmentView', '''AppointmentView
- scanner: Scanner
- appointmentController: AppointmentController
- patientView: PatientView
- dermatologistView: DermatologistView
- treatmentView: TreatmentView
- timeView: TimeView
- invoiceView: InvoiceView
+ showAppointmentMenu(): void
+ displayMenu(): void
+ getUserChoice(): int
+ scheduleAppointment(): void
+ viewAllAppointments(): void
+ updateAppointment(): void
+ cancelAppointment(): void
+ completeAppointment(): void
+ processInvoicePayment(): void
+ searchAppointment(): void''')

uml_diagram.node('PatientView', '''PatientView
+ getPatientDetails(): Patient''')

uml_diagram.node('DermatologistView', '''DermatologistView
- scanner: Scanner
- dermatologistService: DermatologistService
+ selectDermatologist(): DermatologistModel
+ addTimeSlotForDermatologist(): void
+ getValidDateFromUser(): LocalDate
+ getValidStartTimeFromUser(): LocalTime''')

uml_diagram.node('TreatmentView', '''TreatmentView
+ getTreatmentDetails(): Treatment''')

uml_diagram.node('TimeView', '''TimeView
+ getTimeDetails(dermatologist: DermatologistModel): Time''')

uml_diagram.node('InvoiceView', '''InvoiceView
+ getInvoiceDetails(treatment: Treatment): Invoice''')

# Service Layer
uml_diagram.node('AppointmentService', '''AppointmentService
- appointmentRepository: AppointmentRepository
+ createAppointment(appointment: Appointment): void
+ listAppointments(): List<Appointment>
+ updateAppointment(appointment: Appointment): boolean
+ cancelAppointment(appointmentId: int): boolean
+ completeAppointment(appointmentId: int): boolean
+ searchAppointments(criteria: String): List<Appointment>''')

uml_diagram.node('DermatologistService', '''DermatologistService
- dermatologistRepository: DermatologistRepository
+ selectDermatologist(): DermatologistModel
+ addTimeSlotForDermatologist(dermatologist: DermatologistModel, time: Time): boolean
+ getDermatologistAvailability(dermatologistId: int, date: LocalDate): List<Time>''')

uml_diagram.node('PatientService', '''PatientService
- patientRepository: PatientRepository
+ getPatientDetails(patientId: int): Patient
+ createPatient(patient: Patient): void
+ updatePatient(patient: Patient): boolean''')

# Repository Layer
uml_diagram.node('AppointmentRepository', '''AppointmentRepository
- appointments: List<Appointment>
+ save(appointment: Appointment): void
+ findAll(): List<Appointment>
+ findById(appointmentId: int): Optional<Appointment>
+ delete(appointmentId: int): boolean
+ update(appointment: Appointment): boolean''')

uml_diagram.node('DermatologistRepository', '''DermatologistRepository
- dermatologists: List<DermatologistModel>
+ save(dermatologist: DermatologistModel): void
+ findAll(): List<DermatologistModel>
+ findById(id: int): Optional<DermatologistModel>
+ addTimeSlot(dermatologist: DermatologistModel, time: Time): boolean''')

# Domain Models
uml_diagram.node('Appointment', '''Appointment
- appointmentId: int
- patient: Patient
- dermatologist: DermatologistModel
- treatment: Treatment
- time: Time
- invoice: Invoice
- status: AppointmentStatus
+ setStatus(status: AppointmentStatus): void
+ getInvoice(): Invoice
+ getId(): int''')

uml_diagram.node('Patient', '''Patient
- patientId: int
- name: String
- age: int
- contactInfo: String
- medicalHistory: String
+ getId(): int
+ getName(): String
+ getAge(): int''')

uml_diagram.node('DermatologistModel', '''DermatologistModel
- dermatologistId: int
- name: String
- specialization: String
- availableTimeSlots: List<Time>
+ checkAvailability(time: Time): boolean
+ addTimeSlot(time: Time): void''')

uml_diagram.node('Treatment', '''Treatment
- treatmentId: int
- name: String
- description: String
- duration: int
- cost: double
+ getCost(): double
+ getDuration(): int''')

uml_diagram.node('Invoice', '''Invoice
- invoiceId: int
- totalAmount: double
- amountPaid: double
- status: PaymentStatus
+ makePayment(amount: double): void
+ isPaid(): boolean
+ getBalance(): double''')

uml_diagram.node('Time', '''Time
- date: LocalDate
- startTime: LocalTime
- endTime: LocalTime
+ getDate(): LocalDate
+ getStartTime(): LocalTime
+ getEndTime(): LocalTime
+ isAvailable(): boolean''')

# Relationships
# View to Service Layer
uml_diagram.edge('AppointmentView', 'AppointmentService', 'uses')
uml_diagram.edge('AppointmentView', 'PatientView', 'uses')
uml_diagram.edge('AppointmentView', 'DermatologistView', 'uses')
uml_diagram.edge('AppointmentView', 'TreatmentView', 'uses')
uml_diagram.edge('AppointmentView', 'TimeView', 'uses')
uml_diagram.edge('AppointmentView', 'InvoiceView', 'uses')

# Service to Repository Layer
uml_diagram.edge('AppointmentService', 'AppointmentRepository', 'uses')
uml_diagram.edge('DermatologistService', 'DermatologistRepository', 'uses')
uml_diagram.edge('PatientService', 'PatientRepository', 'uses')

# Domain Model Relationships
uml_diagram.edge('Appointment', 'Patient', 'aggregation')
uml_diagram.edge('Appointment', 'DermatologistModel', 'aggregation')
uml_diagram.edge('Appointment', 'Treatment', 'aggregation')
uml_diagram.edge('Appointment', 'Time', 'aggregation')
uml_diagram.edge('Appointment', 'Invoice', 'composition')
uml_diagram.edge('DermatologistModel', 'Time', 'aggregation')

# Repository to Model Relationships
uml_diagram.edge('AppointmentRepository', 'Appointment', 'manages')
uml_diagram.edge('DermatologistRepository', 'DermatologistModel', 'manages')
uml_diagram.edge('PatientRepository', 'Patient', 'manages')

# Generate the diagram
uml_diagram.render(view=True)
