from graphviz import Digraph

# Create a new UML diagram
uml_diagram = Digraph('Complete_UML_Class_Diagram', filename='./UML_Diagram_Complete_Combined3.png', format='png')

# Add model classes (from the first and second code combined)
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

# Adding classes with attributes and methods for the provided code (from the original code block)
classes = {
    "AppointmentView": {
        "attributes": ["scanner: Scanner", "appointmentController: AppointmentController", 
                       "patientView: PatientView", "dermatologistView: DermatologistView", 
                       "treatmentView: TreatmentView", "timeView: TimeView", "invoiceView: InvoiceView"],
        "methods": ["showAppointmentMenu(): void", "displayMenu(): void", "getUserChoice(): int", 
                    "scheduleAppointment(): void", "viewAllAppointments(): void", 
                    "updateAppointment(): void", "cancelAppointment(): void", 
                    "completeAppointment(): void", "processInvoicePayment(): void", 
                    "searchAppointment(): void"]
    },
    "AppointmentController": {
        "attributes": [],
        "methods": ["createAppointment(patient, dermatologist, treatment, time, invoice): Appointment", 
                    "listAppointments(): List<Appointment>", "getAppointmentById(appointmentId: int): Appointment", 
                    "updateAppointment(appointmentId: int, patient, dermatologist, treatment, time): bool", 
                    "cancelAppointmentById(appointmentId: int): bool"]
    },
    "PatientView": {
        "attributes": [],
        "methods": ["getPatientDetails(): Patient"]
    },
    "DermatologistView": {
        "attributes": ["scanner: Scanner", "dermatologistRepository: DermatologistRepository"],
        "methods": ["selectDermatologist(): DermatologistModel", 
                    "addTimeSlotForDermatologist(): void", "getValidDateFromUser(): LocalDate", 
                    "getValidStartTimeFromUser(): LocalTime", "getValidInputFromUser(prompt, parser): <T>"]
    },
    "TreatmentView": {
        "attributes": [],
        "methods": ["getTreatmentDetails(): Treatment"]
    },
    "TimeView": {
        "attributes": [],
        "methods": ["getTimeDetails(dermatologist): Time"]
    },
    "InvoiceView": {
        "attributes": [],
        "methods": ["getInvoiceDetails(treatment): Invoice"]
    },
    "Appointment": {
        "attributes": ["appointmentId: int", "patient: Patient", "dermatologist: DermatologistModel", 
                       "treatment: Treatment", "time: Time", "invoice: Invoice", "status: String"],
        "methods": []
    },
    "Patient": {
        "attributes": ["patientId: int", "name: String", "age: int", "gender: String"],
        "methods": []
    },
    "DermatologistModel": {
        "attributes": ["dermatologistId: int", "name: String", "specialization: String"],
        "methods": ["checkAvailability(time: Time): bool"]
    },
    "Treatment": {
        "attributes": ["treatmentId: int", "name: String", "description: String", "cost: double"],
        "methods": []
    },
    "Time": {
        "attributes": ["date: LocalDate", "startTime: LocalTime", "endTime: LocalTime"],
        "methods": []
    },
    "Invoice": {
        "attributes": ["invoiceId: int", "totalAmount: double", "amountPaid: double", "isPaid: bool"],
        "methods": ["makePayment(amount: double): void", "isPaid(): bool"]
    },
    "Dermatologist": {
        "attributes": ["name: String"],
        "methods": []
    },
    "TimeImpl": {
        "attributes": ["date: LocalDate", "startTime: LocalTime", "endTime: LocalTime"],
        "methods": ["getEndTime(): LocalTime"]
    },
    "DermatologistRepository": {
        "attributes": [],
        "methods": ["addTimeSlot(dermatologist: DermatologistModel, time: Time): bool"]
    }
}

# Adding these classes to the diagram
for class_name, class_info in classes.items():
    attributes = "\l".join(class_info["attributes"]) + "\l"
    methods = "\l".join(class_info["methods"]) + "\l"
    uml_diagram.node(class_name, f"{class_name}|\l{attributes}|{methods}")

# Relationships among classes (from the first and second code combined)
relationships = [
    ("AppointmentView", "AppointmentController", "uses"),
    ("AppointmentView", "PatientView", "uses"),
    ("AppointmentView", "DermatologistView", "uses"),
    ("AppointmentView", "TreatmentView", "uses"),
    ("AppointmentView", "TimeView", "uses"),
    ("AppointmentView", "InvoiceView", "uses"),
    ("Appointment", "Patient", "aggregation"),
    ("Appointment", "DermatologistModel", "aggregation"),
    ("Appointment", "Treatment", "aggregation"),
    ("Appointment", "Time", "aggregation"),
    ("Appointment", "Invoice", "aggregation"),
    ("DermatologistModel", "Dermatologist", "composition"),
    ("DermatologistView", "DermatologistRepository", "uses"),
    ("DermatologistRepository", "DermatologistModel", "aggregation"),
    ("TimeImpl", "Time", "implements"),
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

# Adding relationships to the UML diagram
for src, dst, relationship_type in relationships:
    if relationship_type == "composition":
        uml_diagram.edge(src, dst, arrowhead="diamond")
    elif relationship_type == "aggregation":
        uml_diagram.edge(src, dst, arrowhead="odiamond")
    elif relationship_type == "manages":
        uml_diagram.edge(src, dst, arrowhead="normal", style="dashed")
    else:
        uml_diagram.edge(src, dst)

# Render and save the diagram
uml_diagram.render()
uml_diagram.attr(dpi='300')  # Set the DPI to 300 for better resolution
uml_diagram.attr(rankdir='LR', size='12,8')  # Increase the diagram size for more detail

print("UML diagram saved as 'UML_Diagram_Complete_Combined.png'")
