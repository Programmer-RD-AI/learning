from graphviz import Digraph

# Create a new UML diagram
uml_diagram = Digraph('Complete_UML_Class_Diagram', filename='./UML_Diagram_Updated5.png', format='png')

# Define class nodes with attributes, methods, visibility, and relationships
uml_diagram.node('Appointment', '''Appointment
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

uml_diagram.node('Patient', '''Patient
- patientId: int
- name: String
- age: int
- address: String
- contactInfo: String
+ getPatientDetails(): Patient
''')

uml_diagram.node('DermatologistModel', '''DermatologistModel
- name: String
+ checkAvailability(time: Time): boolean
''')

uml_diagram.node('Treatment', '''Treatment
- treatmentId: int
- treatmentName: String
- cost: double
+ getTreatmentDetails(): Treatment
''')

uml_diagram.node('Invoice', '''Invoice
- invoiceId: int
- totalAmount: double
- amountPaid: double
+ makePayment(amount: double): void
+ isPaid(): boolean
''')

uml_diagram.node('Time', '''Time
- date: LocalDate
- startTime: LocalTime
+ getEndTime(): LocalTime
''')

uml_diagram.node('DermatologistRepository', '''DermatologistRepository
- dermatologists: List<DermatologistModel>
+ addTimeSlot(dermatologist: DermatologistModel, time: Time): boolean
''')

uml_diagram.node('AppointmentRepository', '''AppointmentRepository
- appointments: List<Appointment>
+ createAppointment(appointment: Appointment): void
+ listAppointments(): List<Appointment>
+ getAppointmentById(appointmentId: int): Appointment
+ cancelAppointmentById(appointmentId: int): boolean
+ updateAppointment(appointmentId: int, patient: Patient, dermatologist: DermatologistModel, treatment: Treatment, time: Time): boolean
+ getAppointmentsByPatientName(patientName: String): List<Appointment>
''')

uml_diagram.node('AppointmentService', '''AppointmentService
+ scheduleAppointment(): void
+ viewAllAppointments(): void
+ updateAppointment(): void
+ cancelAppointment(): void
+ completeAppointment(): void
+ searchAppointment(): void
''')

uml_diagram.node('DermatologistService', '''DermatologistService
+ selectDermatologist(): DermatologistModel
+ addTimeSlotForDermatologist(): void
''')

uml_diagram.node('TimeService', '''TimeService
+ getTimeDetails(dermatologist: DermatologistModel): Time
''')

uml_diagram.node('TreatmentService', '''TreatmentService
+ getTreatmentDetails(): Treatment
''')

uml_diagram.node('PatientService', '''PatientService
+ getPatientDetails(): Patient
''')

# Define relationships among classes with labels for each relationship
relationships = [
    ("Appointment", "Patient", "aggregation"),
    ("Appointment", "DermatologistModel", "aggregation"),
    ("Appointment", "Treatment", "aggregation"),
    ("Appointment", "Time", "aggregation"),
    ("Appointment", "Invoice", "aggregation"),
    ("DermatologistModel", "DermatologistRepository", "aggregation"),
    ("AppointmentRepository", "Appointment", "manages"),
    ("DermatologistRepository", "DermatologistModel", "manages"),
    ("DermatologistRepository", "Time", "manages"),
    ("AppointmentService", "AppointmentRepository", "uses"),
    ("AppointmentService", "PatientService", "uses"),
    ("AppointmentService", "DermatologistService", "uses"),
    ("AppointmentService", "TreatmentService", "uses"),
    ("AppointmentService", "TimeService", "uses"),
    ("DermatologistService", "DermatologistRepository", "uses"),
    ("DermatologistService", "TimeService", "uses"),
    ("TreatmentService", "Treatment", "uses"),
    ("PatientService", "Patient", "uses"),
    ("TimeService", "Time", "uses")
]

# Add relationships with labels to the diagram
for src, dst, relationship_type in relationships:
    if relationship_type == "composition":
        uml_diagram.edge(src, dst, arrowhead="diamond", label="composition")
    elif relationship_type == "aggregation":
        uml_diagram.edge(src, dst, arrowhead="odiamond", label="aggregation")
    elif relationship_type == "manages":
        uml_diagram.edge(src, dst, arrowhead="normal", style="dashed", label="manages")
    elif relationship_type == "uses":
        uml_diagram.edge(src, dst, label="uses")

# Render and save the diagram
uml_diagram.render()
uml_diagram.attr(dpi='300')  # High resolution
uml_diagram.attr(rankdir='LR', size='12,8')  # Horizontal layout and size adjustment

print("UML diagram with relationship labels has been generated and saved.")
